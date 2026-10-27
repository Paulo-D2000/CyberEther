# Aligned QPSK → AR4JA LDPC demo

Open **Aligned AR4JA LDPC Decoder** from CyberEther's example list, or open
`examples/flowgraphs/ldpc-aligned.yml`. Running it requires a working Python
runtime with NumPy selected in CyberEther settings.

The graph uses a verified nonzero encoded codeword, fresh QPSK AWGN at an
editable Eb/N0 (default 3 dB), ideal I/Q soft demapping, the native CPU LDPC
decoder, and a known-reference monitor. It plots the noisy constellation and
pre-/post-FEC per-frame BER over the same 2048 transmitted bit positions.
The full decoded output is 2560 U8 values (one byte per hard-decision bit).

The monitor's third output has seven F64 values:

1. Observed frames
2. Pre-FEC bit errors
3. Post-FEC bit errors
4. Compared transmitted bits (frames × 2048)
5. Wrong full codewords
6. Full-codeword frame-error rate
7. Current number of unsatisfied parity checks

Inspect that output for counts. Syndrome weight (7) is **not BER**, and a
zero syndrome is not proof of a correctly recovered message. BER here is
possible only because the reference transmitted bits are known. The history
plots start with zero-filled slots; they are not confidence intervals.

## Aligned file replay

Export the separate benchmark corpus, preserving every frame including
low-SNR failures:

Reference files are already tracked under
`src/domains/dsp/ldpc_decoder/test_data/iq/`; a fresh clone can replay them
without regenerating anything. To regenerate them from the committed corpus:

```
python src/domains/dsp/ldpc_decoder/export_demo_iq.py --output-directory src/domains/dsp/ldpc_decoder/test_data/iq
```

This creates `ldpc-aligned.cf32` (little-endian interleaved I/Q float32,
one complex sample per symbol), `ldpc-reference.u8` (2560 reference bits
per codeword), and a format/hash manifest. There are 64 frames, each with
1024 QPSK symbols representing 2048 transmitted bits. QPSK mapping is I
then Q, positive axis = bit 0. No preamble or framing marker is inserted:
the file layout supplies exact frame boundaries.

Set `IQ_FILE` and `TRUTH_FILE` in the example's channel Python block to the
two absolute file paths, and execute its code. On Windows, use forward
slashes or Python raw strings for paths. Replay wraps at EOF; repeated
passes are not independent recorded observations. Eb/N0 edits affect only
generated noise, not file replay. The constellation displays normalized
IQ, not calibrated radio power.

## Scope and verification

This is a fixed AR4JA rate-1/2 k=1024 decoder experiment, **not a complete
CCSDS receiver**. It assumes correct phase, symbol timing, I/Q bit order and
codeword boundaries. There is no live encoder, oversampled pulse shaping,
carrier/timing recovery, frame synchronization, payload deframer or CRC.
The existing QPSK recovery front end can be connected later with an
appropriate soft-demap/frame-alignment stage. `psk_demod` itself requires
at least 2 SPS; the aligned 1-SPS example bypasses its loops.

`test_demo.py` checks the embedded Python programs headlessly: QPSK mapping,
AWGN variance, file replay/wrap, reference-bit denominators and syndrome
reporting. The native block tests check example registration and parsing.
Neither is a claim of a visually verified GUI or an end-to-end Python
runtime/scheduler test. Generate the example again after editing its
generator with `python src/domains/dsp/ldpc_decoder/make_demo.py`.
