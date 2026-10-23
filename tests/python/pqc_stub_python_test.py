"""M5 phase 1 (D13): the PQC carrier has a read-only Python snapshot class and
no Python construction until phase 7; existing carriers are unaffected."""
import sys
import unittest
import numpy as np
import kan


class PqcStub(unittest.TestCase):
    def test_snapshot_class_is_read_only_and_not_constructible(self):
        self.assertTrue(hasattr(kan, "PqcEdges"))
        with self.assertRaises(TypeError):
            kan.PqcEdges()
        self.assertIsInstance(kan.PqcEdges.coefficients, property)
        self.assertIsNone(kan.PqcEdges.coefficients.fset)

    def test_no_python_construction_before_phase_7(self):
        self.assertFalse(hasattr(kan, "PqcConfig"))
        with self.assertRaises(TypeError):
            kan.Layer(1, 1, pqc=None)

    def test_existing_carriers_are_unaffected(self):
        layer = kan.Layer(2, 1, wavelet=kan.WaveletConfig(2))
        self.assertIsInstance(layer.carrier, kan.WaveletEdges)
        y = layer.forward(np.zeros((3, 2)))
        np.testing.assert_array_equal(y, np.zeros((3, 1)))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
