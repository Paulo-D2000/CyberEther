#!/usr/bin/env python3
"""Generate the aligned-LLR demo from a verified nonzero golden codeword.

The generated .yml uses JSON syntax, a supported YAML subset. No Python
dependencies are needed to generate it; running the flowgraph needs NumPy.
"""
import json
import base64
from pathlib import Path
import re
import struct
import zlib


def generate():
    here = Path(__file__).resolve().parent
    fixture = (here / "test_data/ldpc_golden.hh").read_text()
    data = fixture.split("kCodewords[kFrames][kFrameBits] = {", 1)[1]
    first = re.search(r"\{([^{}]+)\}", data).group(1)
    bits = "".join(re.findall(r"\b[01]\b", first))
    if len(bits) != 2560 or set(bits) != {"0", "1"}:
        raise ValueError("Expected a nonzero verified 2560-bit codeword")

    source = f'''import numpy as np

# Verified encoded nonzero frame, not an encoder implementation.
BITS = "{bits}"
truth = np.fromiter((int(b) for b in BITS), dtype=np.uint8)
rng = np.random.default_rng(42)
EBN0_DB = 3.0  # Edit and run this block to change the channel.
IQ_FILE = ""   # Optional aligned .cf32 file from export_demo_iq.py
TRUTH_FILE = ""  # Matching full-codeword .u8 reference file
replay_index = 0
if bool(IQ_FILE) != bool(TRUTH_FILE):
    raise ValueError("Set both IQ_FILE and TRUTH_FILE, or neither")
if IQ_FILE:
    replay_iq = np.fromfile(IQ_FILE, dtype="<c8").reshape(-1, 1024)
    replay_truth = np.fromfile(TRUTH_FILE, dtype=np.uint8).reshape(-1, 2560)
    if len(replay_iq) == 0 or len(replay_iq) != len(replay_truth):
        raise ValueError("Mismatched or empty IQ/reference files")

def compute(ctx):
    global replay_index
    if IQ_FILE:
        ctx.outputs[0][...] = replay_iq[replay_index]
        ctx.outputs[1][...] = replay_truth[replay_index]
        replay_index = (replay_index + 1) % len(replay_iq)
        return
    # QPSK bit 0 -> positive axis; R=1024/2048=1/2.
    # This stage assumes perfect symbol timing and carrier/frame alignment.
    sigma = np.sqrt(1.0 / (2.0 * 0.5 * 10.0**(EBN0_DB / 10.0)))
    axes = 1.0 - 2.0 * truth[:2048].astype(np.float32)
    axes += sigma * rng.standard_normal(2048).astype(np.float32)
    ctx.outputs[0][...] = (axes[0::2] + 1j * axes[1::2]) / np.sqrt(2.0)
    ctx.outputs[1][...] = truth
    # Plot index only, not a claimed physical sample rate.
    ctx.output_attrs[0]["sampleRate"] = 125000.0
    ctx.output_attrs[0]["frequency"] = 0.0
'''
    demap = '''import numpy as np

def compute(ctx):
    # Known I then Q bit order, known phase, one sample per symbol.
    # These are soft values proportional to LLR, not hard slicing.
    symbols = ctx.inputs[0]
    ctx.outputs[0][0::2] = symbols.real * np.sqrt(2.0)
    ctx.outputs[0][1::2] = symbols.imag * np.sqrt(2.0)
'''
    monitor = '''import numpy as np

history_pre = np.zeros(128, dtype=np.float32)
history_post = np.zeros(128, dtype=np.float32)
frames = pre_errors = post_errors = frame_errors = 0

def compute(ctx):
    global frames, pre_errors, post_errors, frame_errors
    llr, truth, decoded = ctx.inputs
    raw = (llr < 0).astype(np.uint8)
    pre = np.count_nonzero(raw != truth[:2048])
    # Post-FEC comparison uses the SAME 2048 transmitted positions.
    post = np.count_nonzero(decoded[:2048] != truth[:2048])
    wrong_frame = np.any(decoded != truth)
    frames += 1
    pre_errors += int(pre)
    post_errors += int(post)
    frame_errors += int(wrong_frame)
    syndrome = np.bitwise_xor.reduceat(decoded[check_vars], check_offsets[:-1])
    unsatisfied = np.count_nonzero(syndrome)
    history_pre[:-1] = history_pre[1:]
    history_post[:-1] = history_post[1:]
    history_pre[-1] = pre / 2048.0
    history_post[-1] = post / 2048.0
    ctx.outputs[0][...] = history_pre
    ctx.outputs[1][...] = history_post
    # Numeric counters available on this output's wire inspector:
    # [frames, pre errors, post errors, compared bits, wrong frames, FER,
    #  current unsatisfied checks]. Syndrome weight is NOT a BER estimate.
    ctx.outputs[2][...] = [frames, pre_errors, post_errors, frames * 2048,
                         frame_errors, frame_errors / frames, unsatisfied]
    for index in (0, 1):
        ctx.output_attrs[index]["sampleRate"] = 128.0
        ctx.output_attrs[index]["frequency"] = 64.0
'''
    tables = (here / "ldpc_ar4ja_tables.hh").read_text()
    def table_values(name):
        body = tables.split(name, 1)[1].split("= {", 1)[1].split("};", 1)[0]
        return [int(value) for value in re.findall(r"\d+", body)]
    offsets = table_values("kCheckOffsets[")
    columns = table_values("kCheckVar[")
    packed = struct.pack("<" + "H" * (len(offsets) + len(columns)),
                         *(offsets + columns))
    encoded = base64.b64encode(zlib.compress(packed)).decode("ascii")
    monitor = monitor.replace("import numpy as np\n", "import numpy as np\n"
        + "import base64, zlib\n"
        + "# Embedded AR4JA check topology, uint16 little-endian offsets then variables.\n"
        + "topology = np.frombuffer(zlib.decompress(base64.b64decode(\""
        + encoded + "\")), dtype=\"<u2\")\n"
        + "check_offsets = topology[:1537].astype(np.int64)\n"
        + "check_vars = topology[1537:].astype(np.int64)\n", 1)

    def node(name, module, config, x, y, inputs=None, runtime="native"):
        result = dict(name=name, module=module, device="cpu", runtime=runtime,
                      provider="generic", config=config,
                      meta={"node": dict(x=x, y=y, width=360, height=0)})
        if inputs:
            result["input"] = inputs
        return result

    def spec(shape, dtype):
        return dict(shape=f"[{shape}]", axes="[S]", dtype=dtype, device="cpu")

    graph = [node("scope", "note", {"content": """# Aligned CCSDS AR4JA decoder demo

**Requires Python with NumPy. CPU only.**

This is a known encoded nonzero frame repeatedly QPSK-modulated with fresh
AWGN noise. An ideal coherent demapper extracts I/Q soft values. It demonstrates
aligned soft-input decoding, not a complete
CCSDS receiver. There is no live encoder, carrier/timing recovery, frame
synchronizer, CRC or deframer in this example.

Edit EBN0_DB in the channel block and run it (Ctrl+Enter). Default: 3 dB.
For aligned 1-SPS IQ replay, export files with export_demo_iq.py and set
IQ_FILE and TRUTH_FILE in that block. Files loop at the end; counters then
count repeated observations, not additional independent recorded frames.
The decoder is fixed to AR4JA rate 1/2, k=1024; the code is not selectable.
Its output is the full 2560-bit codeword as U8, one byte per decision.

## Plots and counters
The two plots show **per-frame BER over the same 2048 transmitted positions**
before and after decoding. Horizontal axis: the latest 128 frames, initially
zero-filled. Normalized vertical range 0..1 is BER, not dB or radio power.
The monitor's third output is [frames, pre errors, post errors, compared
bits, wrong full codewords, full-codeword FER, current unsatisfied parity
checks]. Inspect that wire for counts. Unsatisfied checks are syndrome
weight, NOT BER. Passing every check does not prove the message is correct.
Zero observed errors is not proof of zero underlying BER. This repeats one
known message; use the separate multi-frame benchmark corpus for timing.

The source clock only triggers compute. Its values are ignored; it does
not simulate carrier or timing offsets. Next stages add actual modulation,
synchronization and a defined payload framing format.
"""}, 0, 420)]
    graph.append(node("clock", "signal_generator", dict(
        signalType="cosine", signalDataType="F32", sampleRate=2048,
        frequency=1, amplitude=1, bufferSize=2048), 0, 0))
    graph.append(node("channel", "python", dict(
        code=source, inputCount=1, outputCount=2,
        outputTensorSpecs=[spec(1024, "CF32"), spec(2560, "U8")],
        throttled=False), 400, 0,
        {"input0": "${graph.clock.output.signal}"}, "python"))
    graph.append(node("soft_demapper", "python", dict(
        code=demap, inputCount=1, outputCount=1,
        outputTensorSpecs=[spec(2048, "F32")], throttled=False), 800, 0,
        {"input0": "${graph.channel.output.output0}"}, "python"))
    graph.append(node("decode", "ldpc_decoder", {"iterations": 50}, 1200, 0,
        {"llr": "${graph.soft_demapper.output.output0}"}))
    graph.append(node("measure", "python", dict(
        code=monitor, inputCount=3, outputCount=3,
        outputTensorSpecs=[spec(128, "F32"), spec(128, "F32"), spec(7, "F64")],
        throttled=False), 1600, 0, {
            "input0": "${graph.soft_demapper.output.output0}",
            "input1": "${graph.channel.output.output1}",
            "input2": "${graph.decode.output.bits}"}, "python"))
    graph.append(node("received_constellation", "constellation", dict(
        xLabel="In-phase", yLabel="Quadrature"), 800, 440,
        {"signal": "${graph.channel.output.output0}"}))
    for name, port, y in (("pre_fec_ber", "output0", 0),
                          ("post_fec_ber", "output1", 440)):
        graph.append(node(name, "lineplot", dict(
            averaging=1, maxHold=False, fill=False, density=False,
            rangeMin=0.0, rangeMax=1.0, xLabel="Recent frame index",
            yLabel="BER (transmitted positions)"), 2000, y,
            {"signal": "${graph.measure.output." + port + "}"}))
    document = dict(version=2, title="Aligned AR4JA LDPC Decoder",
        summary="Known-data CPU decoding with measured pre/post-FEC BER.",
        author="Paulo-D2000", license="MIT",
        description="Fixed-code aligned-LLR demonstration; requires Python and NumPy. No synchronizer or deframer yet.",
        graph=graph)
    destination = here.parents[3] / "examples/flowgraphs/ldpc-aligned.yml"
    destination.write_text(json.dumps(document, indent=2) + "\n")
    return document, destination


if __name__ == "__main__":
    _, destination = generate()
    print(destination)
