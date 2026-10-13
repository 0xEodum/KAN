"""Backlog M3: the SiLU residual branch through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def silu(x):
    return x / (1 + np.exp(-x))


def layer_with_parameters():
    layer = kan.Layer(3, 2, kan.ChebyshevConfig(4))
    layer.set_parameters(np.linspace(-0.3, 0.3, 24).reshape(2, 3, 4), np.array([0.1, -0.2]))
    return layer


class Residual(unittest.TestCase):
    def test_default_absent_and_set_residual(self):
        layer = layer_with_parameters()
        self.assertIsNone(layer.residual)
        w = np.array([[0.5, -0.25, 0.125], [1.0, 0.0, -2.0]])
        layer.set_residual(w)
        np.testing.assert_array_equal(layer.residual, w)
        self.assertEqual(layer.residual.shape, (2, 3))
        snapshot = layer.residual
        snapshot[0, 0] = 9.0  # an owned copy
        np.testing.assert_array_equal(layer.residual, w)
        layer.set_residual(None)
        self.assertIsNone(layer.residual)

    def test_set_residual_is_strict_and_atomic(self):
        layer = layer_with_parameters()
        w = np.ones((2, 3))
        layer.set_residual(w)
        with self.assertRaises(ValueError):
            layer.set_residual(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            layer.set_residual(np.ones(6))
        with self.assertRaises(TypeError):
            layer.set_residual(np.ones((2, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            layer.set_residual([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        bad = np.ones((2, 3))
        bad[1, 2] = np.inf
        with self.assertRaises(ValueError):
            layer.set_residual(bad)
        np.testing.assert_array_equal(layer.residual, w)

    def test_forward_backward_sgd_and_l2(self):
        plain = layer_with_parameters()
        layer = layer_with_parameters()
        w = np.array([[0.5, -0.25, 0.125], [1.0, 0.0, -2.0]])
        layer.set_residual(w)
        x = np.array([[-0.5, 0.2, 0.9], [3.0, -4.0, 0.1]])
        np.testing.assert_allclose(layer.forward(x), plain.forward(x) + silu(x) @ w.T, rtol=1e-14, atol=1e-15)
        u = np.array([[1.0, -0.5], [0.25, 2.0]])
        g = layer.backward(x, u)
        self.assertEqual(g.residual.shape, (2, 3))
        np.testing.assert_allclose(g.residual, u.T @ silu(x), rtol=1e-14)
        self.assertEqual(plain.backward(x, u).residual.shape, (0,))
        layer.sgd(g, 0.1)
        np.testing.assert_allclose(layer.residual, w - 0.1 * g.residual, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            plain.sgd(g, 0.1)  # gradient of a layer with the branch
        value, penalty = layer.regularization(0.2)
        expected = 0.1 * (np.sum(layer.coefficients ** 2) + np.sum(layer.residual ** 2))
        self.assertAlmostEqual(value, expected, places=14)
        np.testing.assert_allclose(penalty.residual, 0.2 * layer.residual, rtol=0, atol=0)

    def test_branch_survives_family_operations_and_networks(self):
        knots = [-1.0] * 4 + [0.0] + [1.0] * 4
        layer = kan.Layer(2, 2, kan.BSplineConfig(3, knots))
        w = np.array([[0.5, -0.5], [0.25, 1.0]])
        layer.set_residual(w)
        kan.insert_knot(layer, 0.5)
        np.testing.assert_array_equal(layer.residual, w)
        net = kan.Network([layer, kan.Layer(2, 1, kan.ChebyshevConfig(3))])
        np.testing.assert_array_equal(net.layers[0].residual, w)
        self.assertIsNone(net.layers[1].residual)
        x = np.array([[2.0, -3.0]])  # outside the spline domain: only the branch carries a gradient
        g = net.backward(x, np.array([[1.0]]))
        self.assertEqual(g.layers[0].residual.shape, (2, 2))
        net.sgd(g, 0.1)

    def test_noise_init_fields(self):
        n = kan.NoiseInit()
        self.assertEqual((n.residual_mean, n.residual_spread), (0.0, 1.0))
        n = kan.NoiseInit(residual_mean=0.5, residual_spread=0.0, seed=3)
        self.assertEqual((n.residual_mean, n.residual_spread), (0.5, 0.0))
        self.assertNotEqual(n, kan.NoiseInit(seed=3))
        layer = kan.Layer(4, 2, kan.ChebyshevConfig(3))
        layer.set_residual(np.zeros((2, 4)))
        kan.initialize(layer, n)
        np.testing.assert_array_equal(layer.residual, np.full((2, 4), 0.25))  # mean / sqrt(inputs)
        kan.initialize(layer, kan.VarianceScaling(seed=1))
        np.testing.assert_array_equal(layer.residual, 0.0)
        with self.assertRaises(ValueError):
            kan.initialize(layer, kan.NoiseInit(residual_spread=-1.0))
        plain = kan.Layer(4, 2, kan.ChebyshevConfig(3))
        kan.initialize(plain, kan.NoiseInit())
        self.assertIsNone(plain.residual)

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_resident_executes_the_branch(self):
        layer = layer_with_parameters()
        w = np.array([[0.5, -0.25, 0.125], [1.0, 0.0, -2.0]])
        layer.set_residual(w)
        network = kan.Network([layer])
        x = np.array([[-0.5, 0.2, 0.9], [3.0, -4.0, 0.1]])
        u = np.array([[1.0, -0.5], [0.25, 2.0]])
        expected = network.backward(x, u)
        for precision, tol in ((kan.Precision.FLOAT64, 1e-12), (kan.Precision.FLOAT32, 1e-4)):
            gpu = kan.ResidentNetwork(network, 4, precision)
            gpu.upload_input(x)
            gpu.upload_output_gradient(u)
            gpu.forward()
            np.testing.assert_allclose(gpu.download_output(), network.forward(x), rtol=tol, atol=tol)
            gpu.backward(0.0)
            g = gpu.download_gradients()
            self.assertEqual(g.layers[0].residual.shape, (2, 3))
            np.testing.assert_allclose(g.layers[0].residual, expected.layers[0].residual, rtol=tol, atol=tol)
            np.testing.assert_allclose(g.input, expected.input, rtol=tol, atol=tol)
            trained = gpu.download_parameters()
            np.testing.assert_allclose(trained.layers[0].residual, w, rtol=0, atol=tol)
            gpu.upload_parameters(trained)
            with self.assertRaises(ValueError):
                gpu.upload_parameters(kan.Network([layer_with_parameters()]))

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
