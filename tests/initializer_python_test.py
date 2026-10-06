"""Backlog M4: explicit initializers through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def deep(config, width=8, depth=4):
    stages, inputs = [], 2
    for layer in range(depth):
        outputs = 1 if layer == depth - 1 else width
        stages += [kan.InputMap(inputs, kan.TanhMap(1.0)), kan.Layer(inputs, outputs, config)]
        inputs = outputs
    return kan.Network(stages)


class Initializers(unittest.TestCase):
    def test_types_and_defaults(self):
        self.assertEqual(kan.Distribution.UNIFORM.name, "UNIFORM")
        v = kan.VarianceScaling()
        self.assertEqual((v.gain, v.distribution, v.seed), (1.0, kan.Distribution.UNIFORM, 0))
        self.assertEqual((v.denominators.bound, v.denominators.radius), (0.5, 1.0))
        n = kan.NoiseInit(scale=0.5, distribution=kan.Distribution.NORMAL, seed=2**64 - 1)
        self.assertEqual((n.scale, n.seed), (0.5, 2**64 - 1))
        self.assertEqual(kan.VarianceScaling(gain=2.0), kan.VarianceScaling(gain=2.0))
        self.assertNotEqual(kan.NoiseInit(seed=1), kan.NoiseInit(seed=2))
        d = kan.DenominatorInit(bound=0.25, radius=2.0)
        self.assertEqual(kan.VarianceScaling(denominators=d).denominators, d)

    def test_layer_initialization_is_deterministic(self):
        layer = kan.Layer(3, 2, kan.ChebyshevConfig(4))
        np.testing.assert_array_equal(layer.coefficients, 0)  # constructors unchanged
        kan.initialize(layer, kan.VarianceScaling(seed=5))
        again = kan.Layer(3, 2, kan.ChebyshevConfig(4))
        kan.initialize(again, kan.VarianceScaling(seed=5))
        np.testing.assert_array_equal(layer.coefficients, again.coefficients)
        self.assertTrue(np.all(layer.coefficients != 0))
        np.testing.assert_array_equal(layer.bias, 0)

    def test_reference_moments(self):
        variance, moments = kan.reference_moments(kan.LegendreConfig(4))
        self.assertAlmostEqual(variance, 1 / 3)
        np.testing.assert_allclose(moments, [1, 1 / 3, 1 / 5, 1 / 7])
        self.assertEqual(kan.layer_seed(0, 0), 0xE220A8397B1DCDAF)  # first SplitMix64 output of 0

    def test_rational_denominators_and_network(self):
        config = kan.RationalConfig()
        config.denominator_policy = kan.DenominatorPolicy.SMOOTH
        net = deep(config)
        kan.initialize(net, kan.NoiseInit(seed=3, denominators=kan.DenominatorInit(0.4, 1.0)))
        for layer in net.layers:
            if isinstance(layer, kan.Layer):
                b = layer.carrier.denominators
                self.assertTrue(np.all(np.abs(b) >= 0.4 / 2 / 2 - 1e-15))
                self.assertTrue(np.all(np.sum(np.abs(b), axis=2) <= 0.4 + 1e-15))
        with self.assertRaises(ValueError):
            kan.initialize(net, kan.VarianceScaling(gain=0.0))
        guarded = kan.RationalConfig()
        guarded.epsilon = 0.9
        layer = kan.Layer(2, 2, guarded)
        with self.assertRaises(ValueError):
            kan.initialize(layer, kan.VarianceScaling())

    def test_deep_chebyshev_trains_from_variance_scaling(self):
        grid = np.linspace(-0.9, 0.9, 8)
        x = np.array([[a, b] for a in grid for b in grid])
        t = (x[:, 0] * x[:, 1])[:, None]

        def train(net):
            for _ in range(500):
                y = net.forward(x)
                net.sgd(net.backward(x, 2 * (y - t) / len(x)), 0.1)
            return float(np.mean((net.forward(x) - t) ** 2))

        zero = train(deep(kan.ChebyshevConfig(5)))
        net = deep(kan.ChebyshevConfig(5))
        kan.initialize(net, kan.VarianceScaling(seed=1))
        self.assertLess(train(net), 0.2 * zero)

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_resident_from_initialized_network(self):
        net = deep(kan.ChebyshevConfig(5))
        kan.initialize(net, kan.VarianceScaling(seed=1, distribution=kan.Distribution.NORMAL))
        x = np.sin(np.arange(20.0)).reshape(10, 2)
        gpu = kan.ResidentNetwork(net, 10)
        gpu.upload_input(x)
        gpu.forward()
        np.testing.assert_allclose(gpu.download_output(), net.forward(x), rtol=1e-11, atol=1e-13)
        kan.initialize(net, kan.NoiseInit(seed=2))
        gpu.upload_parameters(net)
        gpu.forward()
        np.testing.assert_allclose(gpu.download_output(), net.forward(x), rtol=1e-11, atol=1e-13)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
