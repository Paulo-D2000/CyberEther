# Aligned QPSK reference recording

These are tracked test/demo assets, generated from `../ldpc_corpus.hh`:

- `ldpc-aligned.cf32`: 64 × 1024 symbol-rate complex samples, little-endian
  float32 I followed by float32 Q, exactly one sample per symbol.
- `ldpc-reference.u8`: 64 × 2560 unpacked transmitted-codeword reference bits.
- `ldpc-aligned.json`: dimensions, mapping, limitations and SHA-256 hashes.

Frame boundaries and QPSK phase are already known. This is a synthetic
aligned recording, not an over-the-air capture. The references include
punctured positions; the IQ represents only the first 2048 transmitted bits
per codeword. Low-SNR frames are retained even when decoding fails.

From the repository root, regenerate with:

```
python src/domains/dsp/ldpc_decoder/export_demo_iq.py --output-directory src/domains/dsp/ldpc_decoder/test_data/iq
```

Use absolute paths to these two data files in the example's `IQ_FILE` and
`TRUTH_FILE` settings. The metadata describes file layout, not radio sample
timing or an on-air framing standard.
