"""Headless checks for the embedded QPSK source, demapper and BER monitor.

Run with: python src/domains/dsp/ldpc_decoder/test_demo.py
Requires NumPy; does not claim to test the GUI or carrier/timing recovery.
"""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import hashlib
import json
import unittest

import numpy as np

from make_demo import generate


def context(inputs, outputs):
    return SimpleNamespace(inputs=inputs, outputs=outputs,
                           output_attrs=[{} for _ in outputs])


class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document, _ = generate()
        cls.nodes = {node["name"]: node for node in document["graph"]}

    def script(self, name):
        namespace = {}
        exec(self.nodes[name]["config"]["code"], namespace)
        return namespace

    def test_qpsk_soft_order_and_puncturing(self):
        source = self.script("channel")
        source["EBN0_DB"] = float("inf")  # noise-free symbol mapping
        symbols = np.empty(1024, np.complex64)
        truth = np.empty(2560, np.uint8)
        source["compute"](context([np.zeros(2048)], [symbols, truth]))
        soft = np.empty(2048, np.float32)
        self.script("soft_demapper")["compute"](context([symbols], [soft]))
        np.testing.assert_allclose(soft, 1 - 2 * truth[:2048].astype(np.float32),
                                   atol=1e-6)
        self.assertEqual(set(truth), {0, 1})
        self.assertEqual(self.nodes["decode"]["input"]["llr"],
                         "${graph.soft_demapper.output.output0}")
        metrics = [np.empty(128, np.float32), np.empty(128, np.float32),
                   np.empty(7, np.float64)]
        self.script("measure")["compute"](context([soft, truth, truth], metrics))
        np.testing.assert_array_equal(metrics[2], [1, 0, 0, 2048, 0, 0, 0])

    def test_aligned_iq_file_replay_and_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            iq_path = Path(directory) / "aligned.cf32"
            truth_path = Path(directory) / "truth.u8"
            iq = np.array([np.full(1024, 1 + 1j), np.full(1024, -1 - 1j)],
                          dtype="<c8")
            bits = np.array([np.zeros(2560), np.ones(2560)], dtype=np.uint8)
            iq.tofile(iq_path)
            bits.tofile(truth_path)
            code = self.nodes["channel"]["config"]["code"]
            code = code.replace('IQ_FILE = ""', f'IQ_FILE = {str(iq_path)!r}')
            code = code.replace('TRUTH_FILE = ""', f'TRUTH_FILE = {str(truth_path)!r}')
            namespace = {}
            exec(code, namespace)
            outputs = [np.empty(1024, np.complex64), np.empty(2560, np.uint8)]
            for index in (0, 1, 0):
                namespace["compute"](context([], outputs))
                np.testing.assert_array_equal(outputs[0], iq[index])
                np.testing.assert_array_equal(outputs[1], bits[index])

    def test_repository_reference_recording(self):
        root = Path(__file__).parent / "test_data/iq"
        manifest = json.loads((root / "ldpc-aligned.json").read_text())
        for filename, field in (("ldpc-aligned.cf32", "iq_sha256"),
                                ("ldpc-reference.u8", "truth_sha256")):
            self.assertEqual(hashlib.sha256((root / filename).read_bytes()).hexdigest(),
                             manifest[field])
        corpus = Path(__file__).parent / "test_data/ldpc_corpus.hh"
        self.assertEqual(hashlib.sha256(corpus.read_bytes()).hexdigest(),
                         manifest["corpus_sha256"])
        iq = np.fromfile(root / "ldpc-aligned.cf32", dtype="<c8").reshape(-1, 1024)
        truth = np.fromfile(root / "ldpc-reference.u8", dtype=np.uint8).reshape(-1, 2560)
        self.assertEqual(iq.shape[0], 64)
        self.assertEqual(truth.shape[0], 64)
        self.assertTrue(np.isfinite(iq).all())
        self.assertTrue(np.isin(truth, [0, 1]).all())
        monitor = self.script("measure")
        syndrome = np.bitwise_xor.reduceat(truth[:, monitor["check_vars"]],
                                          monitor["check_offsets"][:-1], axis=1)
        self.assertEqual(np.count_nonzero(syndrome), 0)

    def test_awgn_is_fresh_and_rate_aware(self):
        source = self.script("channel")
        demapper = self.script("soft_demapper")
        errors = []
        for _ in range(8):
            symbols = np.empty(1024, np.complex64)
            truth = np.empty(2560, np.uint8)
            source["compute"](context([], [symbols, truth]))
            soft = np.empty(2048, np.float32)
            demapper["compute"](context([symbols], [soft]))
            errors.append(soft - (1 - 2 * truth[:2048].astype(np.float32)))
        expected_sigma = np.sqrt(1 / 10**(source["EBN0_DB"] / 10))
        self.assertAlmostEqual(np.std(errors), expected_sigma, delta=0.03)
        self.assertFalse(np.array_equal(errors[0], errors[1]))

    def test_ber_denominators_and_full_frame_errors(self):
        monitor = self.script("measure")
        truth = np.zeros(2560, np.uint8)
        soft = np.ones(2048, np.float32)
        soft[:2] = -1
        decoded = truth.copy()
        decoded[:3] = 1
        decoded[-1] = 1  # punctured bit counts for FER, not transmitted BER
        outputs = [np.empty(128, np.float32), np.empty(128, np.float32),
                   np.empty(7, np.float64)]
        ctx = context([soft, truth, decoded], outputs)
        monitor["compute"](ctx)
        self.assertEqual(outputs[0][-1], 2 / 2048)
        self.assertEqual(outputs[1][-1], 3 / 2048)
        np.testing.assert_array_equal(outputs[2][:6], [1, 2, 3, 2048, 1, 1])
        self.assertGreater(outputs[2][6], 0)
        soft[:] = 1
        decoded[:] = 0
        monitor["compute"](ctx)
        np.testing.assert_array_equal(outputs[2], [2, 2, 3, 4096, 1, 0.5, 0])


if __name__ == "__main__":
    unittest.main()
