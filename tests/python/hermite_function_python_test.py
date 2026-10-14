"""Backlog M5: normalized Hermite functions through the NumPy bindings."""
import math
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv
FUNCTION = kan.HermiteNormalization.FUNCTION


def psi_reference(x, terms):
    """psi_n(x) from numpy's physicists' Hermite series, normalized afterwards."""
    h = np.polynomial.hermite.hermvander(x, terms - 1)
    norm = np.array([math.sqrt(2.0**n * math.factorial(n) * math.sqrt(math.pi)) for n in range(terms)])
    return h * np.exp(-np.asarray(x)[..., None]**2 / 2) / norm


class HermiteFunctions(unittest.TestCase):
    def test_configuration(self):
        self.assertEqual({n.name for n in kan.HermiteNormalization}, {"PHYSICISTS", "FUNCTION"})
        default = kan.HermiteConfig(6)
        self.assertEqual(default.normalization, kan.HermiteNormalization.PHYSICISTS)
        function = kan.HermiteConfig(6, FUNCTION)
        self.assertEqual(function, kan.HermiteConfig(size=6, normalization=FUNCTION))
        self.assertNotEqual(function, default)
        default.normalization = FUNCTION
        self.assertEqual(default, function)
        layer = kan.Layer(2, 1, function)
        self.assertEqual(layer.carrier.basis, function)
        self.assertEqual(layer.terms, 6)

    def test_values_derivatives_and_orthonormality(self):
        terms = 20
        for x in [-4.2, -1.0, 0.0, 0.3, 2.7, 6.0]:
            values, derivatives = kan.evaluate_basis(kan.HermiteConfig(terms, FUNCTION), x)
            np.testing.assert_allclose(values, psi_reference(np.array([x]), terms)[0], rtol=1e-12, atol=1e-15)
            h = 1e-6
            plus, _ = kan.evaluate_basis(kan.HermiteConfig(terms, FUNCTION), x + h)
            minus, _ = kan.evaluate_basis(kan.HermiteConfig(terms, FUNCTION), x - h)
            np.testing.assert_allclose(derivatives, (plus - minus) / (2 * h), atol=2e-8)
        # Gauss-Hermite quadrature (weight exp(-x^2)) is exact for psi_m psi_n exp(x^2),
        # a polynomial of degree m+n < 2*nodes.
        terms, nodes = 40, 60
        x, w = np.polynomial.hermite.hermgauss(nodes)
        psi = np.array([kan.evaluate_basis(kan.HermiteConfig(terms, FUNCTION), float(v))[0] for v in x])
        gram = (psi * (w * np.exp(x**2))[:, None]).T @ psi
        np.testing.assert_allclose(gram, np.eye(terms), atol=1e-12)

    def test_reference_moments(self):
        variance, moments = kan.reference_moments(kan.HermiteConfig(5, FUNCTION))
        self.assertEqual(variance, 0.5)
        expected = [math.gamma(n + 0.5) / (math.sqrt(2) * math.pi * math.factorial(n)) for n in range(5)]
        np.testing.assert_allclose(moments, expected, rtol=1e-14)

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_resident_parity_and_upload_validation(self):
        x = np.array([[-2.5, 0.4], [1.1, -0.3], [3.9, 7.5]])
        upstream = np.array([[0.3], [-0.5], [0.2]])
        cpu = kan.Network([kan.Layer(2, 3, kan.HermiteConfig(24, FUNCTION)), kan.Layer(3, 1, kan.HermiteConfig(6, FUNCTION))])
        kan.initialize(cpu, kan.VarianceScaling(seed=3))
        for precision, tolerance in [(kan.Precision.FLOAT64, 1e-12), (kan.Precision.FLOAT32, 1e-4)]:
            gpu = kan.ResidentNetwork(cpu, 4, precision)
            gpu.upload_input(x)
            gpu.upload_output_gradient(upstream)
            gpu.forward()
            np.testing.assert_allclose(gpu.download_output(), cpu.forward(x), atol=tolerance)
            gpu.backward()
            expected = cpu.backward(x, upstream)
            actual = gpu.download_gradients()
            np.testing.assert_allclose(actual.input, expected.input, atol=tolerance)
            for a, e in zip(actual.layers, expected.layers):
                np.testing.assert_allclose(a.coefficients, e.coefficients, atol=tolerance)
        physicists = kan.Network([kan.Layer(2, 3, kan.HermiteConfig(24)), kan.Layer(3, 1, kan.HermiteConfig(6, FUNCTION))])
        gpu = kan.ResidentNetwork(cpu, 4)
        with self.assertRaises(ValueError):
            gpu.upload_parameters(physicists)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
