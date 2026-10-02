"""Backlog M1: typed input maps through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def layer_norm(features, affine=True):
    if not affine:
        return kan.LayerNormMap(epsilon=1e-3)
    gain = 1 + 0.3 * np.sin(np.arange(features))
    bias = 0.2 * np.cos(np.arange(features))
    return kan.LayerNormMap(epsilon=1e-3, gain=list(gain), bias=list(bias))


def seeded(layer, phase):
    c = 0.2 * np.sin(phase + 0.83 * np.arange(layer.coefficients.size)).reshape(layer.coefficients.shape)
    layer.set_parameters(c, np.full(layer.outputs, 0.05))
    return layer


def network():
    return kan.Network([
        kan.InputMap(2, kan.AffineMap(scale=[0.02, -0.01], shift=[0.1, 0.0])),
        seeded(kan.Layer(2, 3, kan.ChebyshevConfig(4)), 0.1),
        kan.InputMap(3, layer_norm(3)),
        seeded(kan.Layer(3, 1, kan.BSplineConfig(2, [-3, -3, -3, 0, 3, 3, 3])), 0.4),
    ])


class InputMaps(unittest.TestCase):
    def test_map_values_snapshots_and_equality(self):
        x = np.array([[0.5, -2.0], [1.0, 4.0]])
        affine = kan.InputMap(2, kan.AffineMap(scale=[2.0, -0.5], shift=[0.25, 1.0]))
        np.testing.assert_array_equal(affine.forward(x), [[1.25, 2.0], [2.25, -1.0]])
        self.assertEqual((affine.inputs, affine.outputs, affine.features), (2, 2, 2))
        self.assertEqual(affine.map, kan.AffineMap(scale=[2.0, -0.5], shift=[0.25, 1.0]))
        self.assertIsInstance(affine.map, kan.AffineMap)
        snapshot = affine.map
        snapshot.scale = [9.0, 9.0]
        self.assertEqual(affine.map.scale, [2.0, -0.5])
        tanh = kan.InputMap(2, kan.TanhMap(scale=0.3))
        np.testing.assert_allclose(tanh.forward(x), np.tanh(0.3 * x), rtol=1e-15)
        g = tanh.backward(x, np.ones_like(x))
        np.testing.assert_allclose(g.input, 0.3 / np.cosh(0.3 * x) ** 2, rtol=1e-14)
        self.assertEqual(g.gain.shape, (0,))
        ln = kan.InputMap(2, kan.LayerNormMap())
        self.assertEqual(ln.map, kan.LayerNormMap(epsilon=1e-5))
        y = ln.forward(x)
        mean = x.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(y, (x - mean) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5), rtol=1e-14)
        ln.set_map(layer_norm(2))
        self.assertEqual(len(ln.map.gain), 2)

    def test_layer_norm_vjps_match_finite_differences(self):
        features, batch, h = 4, 3, 1e-6
        m = kan.InputMap(features, layer_norm(features))
        x = 2 * np.sin(np.arange(batch * features, dtype=float)).reshape(batch, features)
        u = np.cos(np.arange(batch * features, dtype=float)).reshape(batch, features)
        g = m.backward(x, u)
        self.assertEqual(g.input.shape, (batch, features))
        self.assertEqual(g.gain.shape, (features,))
        for index in np.ndindex(x.shape):
            p, q = x.copy(), x.copy()
            p[index] += h
            q[index] -= h
            fd = (np.sum(m.forward(p) * u) - np.sum(m.forward(q) * u)) / (2 * h)
            self.assertAlmostEqual(g.input[index], fd, delta=1e-7)
        base = m.map
        for name in ("gain", "bias"):
            for i in range(features):
                values = list(getattr(base, name))
                plus, minus = list(values), list(values)
                plus[i] += h
                minus[i] -= h
                fp = kan.InputMap(features, kan.LayerNormMap(1e-3, **{**{"gain": base.gain, "bias": base.bias}, name: plus}))
                fm = kan.InputMap(features, kan.LayerNormMap(1e-3, **{**{"gain": base.gain, "bias": base.bias}, name: minus}))
                fd = (np.sum(fp.forward(x) * u) - np.sum(fm.forward(x) * u)) / (2 * h)
                self.assertAlmostEqual(getattr(g, name)[i], fd, delta=1e-7)
        before = m.map
        m.sgd(g, 0.1)
        np.testing.assert_allclose(m.map.gain, np.array(before.gain) - 0.1 * g.gain, rtol=1e-15)

    def test_strict_validation(self):
        with self.assertRaises(ValueError):
            kan.InputMap(2, kan.TanhMap(scale=0.0))
        with self.assertRaises(ValueError):
            kan.InputMap(2, kan.AffineMap(scale=[1.0], shift=[0.0]))
        with self.assertRaises(ValueError):
            kan.InputMap(2, kan.LayerNormMap(gain=[1.0, 1.0]))
        m = kan.InputMap(2, kan.TanhMap())
        with self.assertRaises(TypeError):
            m.forward(np.zeros((1, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            m.forward(np.zeros((1, 3)))
        other = kan.InputMap(2, layer_norm(2))
        with self.assertRaises(ValueError):
            m.sgd(other.backward(np.ones((1, 2)), np.ones((1, 2))), 0.1)

    def test_affine_helpers(self):
        samples = np.array([[-200.0, 5.0], [600.0, 5.0], [200.0, 5.0]])
        affine = kan.affine_from_range(samples, lower=-1.0, upper=1.0)
        y = kan.InputMap(2, affine).forward(samples)
        np.testing.assert_allclose(y[:, 0], [-1, 1, 0], atol=1e-15)
        np.testing.assert_allclose(y[:, 1], 0, atol=1e-15)
        s = kan.InputMap(2, kan.affine_from_moments(samples)).forward(samples)
        np.testing.assert_allclose(s[:, 0].mean(), 0, atol=1e-14)
        np.testing.assert_allclose(s[:, 0].std(), 1, atol=1e-14)
        with self.assertRaises(ValueError):
            kan.affine_from_range(samples, lower=1.0, upper=1.0)

    def test_heterogeneous_network_round_trip(self):
        model = network()
        kinds = [type(stage) for stage in model.layers]
        self.assertEqual(kinds, [kan.InputMap, kan.Layer, kan.InputMap, kan.Layer])
        x = 50 * np.sin(np.arange(6, dtype=float)).reshape(3, 2)
        manual = x
        for stage in model.layers:
            manual = stage.forward(manual)
        np.testing.assert_array_equal(model.forward(x), manual)
        u = np.ones((3, 1))
        g = model.backward(x, u)
        self.assertEqual([type(layer) for layer in g.layers],
                         [kan.InputMapGradients, kan.LayerGradients, kan.InputMapGradients, kan.LayerGradients])
        self.assertEqual(g.input.shape, (3, 2))
        self.assertEqual(g.layers[2].gain.shape, (3,))
        value, penalty = model.regularization(0.5)
        self.assertGreater(value, 0)
        np.testing.assert_array_equal(penalty.layers[2].gain, np.zeros(3))
        with self.assertRaises(ValueError):
            model.insert_knot(0, 0.5)
        before = model.forward(x)
        model.sgd(g, 0.05)
        self.assertFalse(np.array_equal(model.forward(x), before))
        model.insert_knot(3, 1.0)
        with self.assertRaises(ValueError):
            model.sgd(g, 0.05)  # stale topology after refinement
        with self.assertRaises(ValueError):
            kan.Network([kan.InputMap(3, kan.TanhMap()), kan.Layer(2, 1, kan.ChebyshevConfig(3))])

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_resident_parity(self):
        cpu = network()
        x = 40 * np.sin(np.arange(8, dtype=float)).reshape(4, 2)
        u = np.cos(np.arange(4, dtype=float)).reshape(4, 1)
        gpu = kan.ResidentNetwork(cpu, 4)
        gpu.upload_input(x)
        gpu.upload_output_gradient(u)
        for _ in range(2):
            gpu.forward()
            np.testing.assert_allclose(gpu.download_output(), cpu.forward(x), atol=1e-12)
            gpu.backward()
            gg, cg = gpu.download_gradients(), cpu.backward(x, u)
            np.testing.assert_allclose(gg.input, cg.input, atol=1e-12)
            for a, b in zip(gg.layers, cg.layers):
                self.assertIs(type(a), type(b))
                if isinstance(a, kan.InputMapGradients):
                    np.testing.assert_allclose(a.gain, b.gain, atol=1e-12)
                    np.testing.assert_allclose(a.bias, b.bias, atol=1e-12)
                else:
                    np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-12)
            gpu.sgd(0.05)
            cpu.sgd(cg, 0.05)
        trained = gpu.download_parameters().layers
        self.assertIsInstance(trained[0], kan.InputMap)
        self.assertEqual(trained[0].map, cpu.layers[0].map)
        np.testing.assert_allclose(trained[2].map.gain, cpu.layers[2].map.gain, atol=1e-12)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
