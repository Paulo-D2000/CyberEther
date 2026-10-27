#!/usr/bin/env python3
"""Export the benchmark corpus as aligned 1-SPS QPSK IQ plus reference bits.

Usage: python export_demo_iq.py --output-directory /existing/output/directory
No carrier offset, timing offset, preamble or framing markers are inserted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def array_rows(text, name):
    body = text.split(name, 1)[1].split("= {", 1)[1].split("\n};", 1)[0]
    return [match.group(1) for match in re.finditer(r"\{([^{}]+)\}", body)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_directory.is_dir():
        parser.error("output directory must already exist")
    fixture = Path(__file__).parent / "test_data/ldpc_corpus.hh"
    text = fixture.read_text()
    llrs = np.array([[float.fromhex(value.strip().removesuffix("f"))
                      for value in row.split(",") if value.strip()]
                     for row in array_rows(text, "kLlrs[kFrames][kFrameLlrs]")],
                    dtype=np.float32)
    truth = np.array([[int(value) for value in row.split(",") if value.strip()]
                      for row in array_rows(text, "kCodewords[kFrames][kFrameBits]")],
                     dtype=np.uint8)
    if llrs.shape != (64, 2048) or truth.shape != (64, 2560):
        raise ValueError("Unexpected corpus dimensions")
    iq = ((llrs[:, 0::2] + 1j * llrs[:, 1::2]) / np.sqrt(2.0)).astype("<c8")
    iq_path = args.output_directory / "ldpc-aligned.cf32"
    bits_path = args.output_directory / "ldpc-reference.u8"
    iq.tofile(iq_path)
    truth.tofile(bits_path)
    manifest = {
        "description": "Aligned QPSK symbol-rate replay, not an RF recording",
        "iq_format": "little-endian complex float32, interleaved I/Q, 1 SPS",
        "frames": 64, "symbols_per_frame": 1024,
        "reference_format": "U8 hard bits, full 2560-bit codeword per frame",
        "bit_mapping": "I then Q, positive axis = 0; soft values = sqrt(2)*[I,Q]",
        "puncturing": "transmit bits 0..2047; decoder inserts zeros at 2048..2559",
        "corpus_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "iq_sha256": hashlib.sha256(iq_path.read_bytes()).hexdigest(),
        "truth_sha256": hashlib.sha256(bits_path.read_bytes()).hexdigest(),
        "limitations": "Exact frame boundaries supplied by file layout; no synchronization or deframing; finite precision IQ round-trip may perturb failed-frame reference decisions",
    }
    manifest_path = args.output_directory / "ldpc-aligned.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {iq_path}, {bits_path}, {manifest_path}")


if __name__ == "__main__":
    main()
