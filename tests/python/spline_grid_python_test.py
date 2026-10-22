"""Backlog M6: per-input B-spline grids, quantile refit and propagated grid
adaptation through the NumPy interface."""
import sys
import unittest
import numpy as np
import kan


def clamped(p, lo, hi, interior):
    return [lo] * (p + 1) + list(interior) + [hi] * (p + 1)


def grids(p=3, inputs=2):
    rows = [clamped(p, -1, 1, [-.5, 0, .5]), clamped(p, -2, .5, [-1.4, -.9, .1])]
    return kan.BSplineConfig(p, sum(rows[:inputs], []), inputs)


def seeded(layer, scale=.3):
    k = np.arange(layer.coefficients.size, dtype=float).reshape(layer.coefficients.shape)
    layer.set_parameters(scale * np.sin(.37 * k + .3), .1 * np.sin(1.3 * np.arange(layer.outputs, dtype=float)))
    return layer


def samples(batch, inputs=2):
    b = np.arange(batch, dtype=float)[:, None]
    i = np.arange(inputs, dtype=float)[None, :]
    u = .5 + .49 * np.sin(1.7 * b + 2.1 * i)
    lo, hi = np.array([-1., -2.])[:inputs], np.array([1., .5])[:inputs]
    return np.ascontiguousarray(lo + (hi - lo) * u)


class SplineGrid(unittest.TestCase):
    def test_configuration(self):
        c = grids()
        self.assertEqual(c.grids, 2)
        self.assertEqual(c.size, 7)
        self.assertEqual(kan.BSplineConfig(3, [-1] * 4 + [1] * 4).grids, 1)
        self.assertEqual(kan.spline_grid(c, 1).knots, clamped(3, -2, .5, [-1.4, -.9, .1]))
        shared = kan.spline_grid(c, 0)
        self.assertEqual(kan.per_input_grids(shared, 2).grids, 2)
        with self.assertRaises(ValueError): kan.evaluate_basis(c, .1)
        with self.assertRaises(ValueError): kan.Layer(3, 1, c)
        self.assertEqual(kan.Layer(2, 1, c).terms, 7)

    def test_refit_adapt_and_network(self):
        layer = seeded(kan.Layer(2, 3, grids()))
        x = samples(40)
        before = layer.forward(x)
        knots = kan.adapt_grid(layer, x)
        self.assertEqual(knots.shape, (2,))
        np.testing.assert_allclose(layer.forward(x), before, atol=1e-12)
        kan.insert_knots(layer, np.array([.2, -.3]))
        self.assertEqual(layer.terms, 9)
        options = kan.GridRefit(intervals=4, grid_eps=.02, ridge=1e-8, per_input=True)
        self.assertEqual(kan.GridRefit().grid_eps, .02)
        kan.refit_grid(layer, x, options)
        self.assertEqual(layer.terms, 7)
        self.assertEqual(layer.carrier.basis.grids, 2)
        lo, hi = x[:, 0].min(), x[:, 0].max()
        self.assertEqual(layer.carrier.basis.knots[0], lo - .01 * (hi - lo))  # default margin 0.01
        with self.assertRaises(ValueError): kan.refit_grid(layer, np.full((5, 2), .5), options)
        net = kan.Network([seeded(kan.Layer(2, 2, kan.ChebyshevConfig(3)), .4), seeded(kan.Layer(2, 1, grids()))])
        manual = net.layers[1]
        kan.refit_grid(manual, net.layers[0].forward(x))
        self.assertEqual(net.refit_grids(x), 1)
        self.assertEqual(net.layers[1].carrier.basis, manual.carrier.basis)
        k = net.adapt_grid(1, x)
        self.assertEqual(k.shape, (2,))

    @unittest.skipUnless(kan.cuda_enabled and '--cuda' in sys.argv and kan.cuda_available(), 'CUDA build and device')
    def test_resident_refit(self):
        net = kan.Network([seeded(kan.Layer(2, 3, grids())), seeded(kan.Layer(3, 1, kan.per_input_grids(kan.spline_grid(grids(), 0), 3)))])
        x = samples(64)
        for precision, tolerance in [(kan.Precision.FLOAT64, 1e-10), (kan.Precision.FLOAT32, 2e-3)]:
            r = kan.ResidentNetwork(net, 64, precision)
            r.upload_input(x)
            cpu = r.download_parameters()
            r.refit_grids(kan.GridRefit(grid_eps=.1))
            self.assertEqual(cpu.refit_grids(x, kan.GridRefit(grid_eps=.1)), 2)
            refit = r.download_parameters()
            np.testing.assert_allclose(refit.layers[0].carrier.basis.knots, cpu.layers[0].carrier.basis.knots, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(refit.forward(x), cpu.forward(x), rtol=tolerance, atol=tolerance)
            with self.assertRaises(ValueError): r.refit_grids(kan.GridRefit(intervals=9))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
