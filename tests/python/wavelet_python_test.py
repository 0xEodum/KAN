"""Backlog M7: per-edge trainable Mexican-hat wavelets through the NumPy interface."""
import math
import sys
import unittest
import numpy as np
import kan

A = 2 / (math.sqrt(3) * math.pi ** 0.25)


def closed(x, tau, l):
    s = math.exp(l)
    q = (x - tau) / s
    e = math.exp(-q * q / 2)
    a = A / math.sqrt(s)
    return a * (1 - q * q) * e, a / s * q * (q * q - 3) * e, a * e * (-q ** 4 + 3.5 * q * q - .5)


def layer(inputs=2, outputs=3, size=2):
    l = kan.Layer(inputs, outputs, wavelet=kan.WaveletConfig(size))
    shape = (outputs, inputs, size)
    n = inputs * outputs * size
    k = np.arange(n, dtype=float).reshape(shape)
    kan.set_wavelet_parameters(l, .4 * np.sin(.9 * k + .3), .8 * np.sin(1.7 * k + .1), .4 * np.sin(2.3 * k + .2),
                               .1 * np.sin(1.1 * np.arange(outputs, dtype=float)))
    return l


class Wavelet(unittest.TestCase):
    def test_config_evaluation_and_placement(self):
        c = kan.WaveletConfig()
        self.assertEqual(c.size, 1)
        self.assertEqual(kan.WaveletConfig(3), kan.WaveletConfig(3))
        with self.assertRaises(ValueError): kan.Layer(1, 1, wavelet=kan.WaveletConfig(0))
        for x, tau, l in [(.3, -.2, .1), (-1.4, .5, -.7), (2., 0., 0.)]:
            v, dx, dtau, dl = kan.evaluate_wavelet(x, tau, l)
            e = closed(x, tau, l)
            self.assertAlmostEqual(v, e[0], places=13)
            self.assertAlmostEqual(dx, e[1], places=12)
            self.assertEqual(dtau, -dx)
            self.assertAlmostEqual(dl, e[2], places=12)
        self.assertTrue(kan.valid_log_scale(0.0))
        self.assertFalse(kan.valid_log_scale(800.0))
        with self.assertRaises(ValueError): kan.evaluate_wavelet(0., 0., 800.)
        l = kan.Layer(2, 3, wavelet=kan.WaveletConfig(4))
        self.assertIsInstance(l.carrier, kan.WaveletEdges)
        self.assertEqual(l.carrier.config, kan.WaveletConfig(4))
        self.assertEqual(l.carrier.translations.shape, (3, 2, 4))
        np.testing.assert_array_equal(l.carrier.translations[1, 0], [-.75, -.25, .25, .75])
        np.testing.assert_array_equal(l.carrier.log_scales, np.full((3, 2, 4), -math.log(4)))
        np.testing.assert_array_equal(l.coefficients, np.zeros((3, 2, 4)))
        self.assertEqual(l.terms, 4)

    def test_forward_vjps_and_owned_snapshots(self):
        l = layer()
        x = np.array([[.3, -.8], [1.2, -.1], [.05, 2.]])
        u = np.array([[.2, -.3, .5], [.1, .4, -.2], [-.6, .3, .2]])
        e = l.carrier
        y = l.forward(x)
        expected = np.tile(l.bias, (3, 1))
        for b in range(3):
            for o in range(3):
                for i in range(2):
                    for k in range(2):
                        expected[b, o] += e.coefficients[o, i, k] * closed(x[b, i], e.translations[o, i, k], e.log_scales[o, i, k])[0]
        np.testing.assert_allclose(y, expected, atol=1e-14)
        e.translations[:] = 99
        self.assertNotEqual(l.carrier.translations[0, 0, 0], 99)
        g = l.backward(x, u)
        self.assertEqual(g.translations.shape, (3, 2, 2))
        self.assertEqual(g.log_scales.shape, (3, 2, 2))
        self.assertEqual(g.denominators.shape, (0,))
        for key in ['coefficients', 'translations', 'log_scales']:
            for index in np.ndindex(3, 2, 2):
                values = []
                for sign in [1, -1]:
                    c, t, s = l.coefficients, l.carrier.translations, l.carrier.log_scales
                    {'coefficients': c, 'translations': t, 'log_scales': s}[key][index] += sign * 1e-6
                    q = layer(); kan.set_wavelet_parameters(q, c, t, s, l.bias)
                    values.append(float((q.forward(x) * u).sum()))
                self.assertAlmostEqual(getattr(g, key)[index], (values[0] - values[1]) / 2e-6, places=7)
        b = kan.Layer(1, 1, kan.ChebyshevConfig())
        self.assertEqual(b.backward(np.array([[.1]]), np.array([[1.]])).translations.shape, (0,))

    def test_validation_sgd_and_training(self):
        l = layer(1, 1, 2)
        with self.assertRaises(ValueError):
            kan.set_wavelet_parameters(l, l.coefficients, l.carrier.translations, np.full((1, 1, 2), 800.), l.bias)
        with self.assertRaises(ValueError):
            kan.set_wavelet_parameters(l, l.coefficients, np.zeros((1, 1, 3)), l.carrier.log_scales, l.bias)
        with self.assertRaises(TypeError):
            kan.set_wavelet_parameters(l, l.coefficients.astype(np.float32), l.carrier.translations, l.carrier.log_scales, l.bias)
        with self.assertRaises(ValueError):
            kan.set_wavelet_parameters(kan.Layer(1, 1, kan.ChebyshevConfig(2)), np.zeros((1, 1, 2)), np.zeros((1, 1, 2)),
                                       np.zeros((1, 1, 2)), np.zeros(1))
        # One wavelet learns an off-center bump: translation and scale move to it.
        l = kan.Layer(1, 1, wavelet=kan.WaveletConfig(1))
        kan.set_wavelet_parameters(l, np.full((1, 1, 1), .5), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros(1))
        x = np.linspace(-2, 2, 81).reshape(-1, 1)
        target = 1.3 * np.array([closed(v, .6, -.5)[0] for v in x[:, 0]]).reshape(-1, 1)
        for _ in range(3000):
            l.sgd(l.backward(x, 2 * (l.forward(x) - target) / len(x)), .2)
        self.assertLess(float(np.mean((l.forward(x) - target) ** 2)), 1e-6)
        self.assertAlmostEqual(float(l.carrier.translations[0, 0, 0]), .6, places=2)
        self.assertAlmostEqual(float(l.carrier.log_scales[0, 0, 0]), -.5, places=2)
        g = l.backward(x, np.full_like(x, 1e30))
        with self.assertRaises(OverflowError): l.sgd(g, 1.)

    @unittest.skipUnless(kan.cuda_enabled and '--cuda' in sys.argv and kan.cuda_available(), 'CUDA build and device')
    def test_resident_parity_and_upload(self):
        net = kan.Network([layer(2, 3, 2), layer(3, 1, 1)])
        x = np.linspace(-1, 1, 10).reshape(5, 2)
        u = np.linspace(-.5, .5, 5).reshape(5, 1)
        for precision, tolerance in [(kan.Precision.FLOAT64, 1e-11), (kan.Precision.FLOAT32, 2e-4)]:
            r = kan.ResidentNetwork(net, 5, precision)
            r.upload_input(x); r.upload_output_gradient(u); r.forward(); r.backward()
            np.testing.assert_allclose(r.download_output(), net.forward(x), rtol=tolerance, atol=tolerance)
            g, e = r.download_gradients(), net.backward(x, u)
            np.testing.assert_allclose(g.layers[0].translations, e.layers[0].translations, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(g.layers[0].log_scales, e.layers[0].log_scales, rtol=tolerance, atol=tolerance)
            r.sgd(.01)
            trained = r.download_parameters()
            self.assertIsInstance(trained.layers[0].carrier, kan.WaveletEdges)
            r.upload_parameters(net)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
