"""Independent localized mathematics, strict bindings, training and GPU parity."""
import sys
import unittest
import numpy as np
import kan


def config(kind):
    b = kan.BasisConfig()
    b.kind, b.size = kind, 4
    b.degree, b.knots = 3, [0.] * 4 + [1.] * 4
    b.centers, b.scales = [0., .3, .6, 1.], [.2, .3, .4, .5]
    b.trainable_rbf = kind == kan.BasisKind.GaussianRbf
    b.log_widths = [-1., -.8, -.6, -.4]
    return b


class M3(unittest.TestCase):
    def test_new_basis_contracts(self):
        b = config(kan.BasisKind.BSpline)
        v, d = kan.evaluate_basis(b, .5)
        np.testing.assert_allclose(v, [.125, .375, .375, .125])
        np.testing.assert_allclose(d, [-.75, -.75, .75, .75])
        v, d = kan.evaluate_basis(config(kan.BasisKind.MexicanHat), .2)
        self.assertTrue(np.isfinite(v).all() and np.isfinite(d).all())

    def test_nonlinear_vjp_and_snapshot(self):
        l = kan.Layer(1, 1, config(kan.BasisKind.GaussianRbf))
        l.set_parameters(np.array([[[.2, -.3, .4, .1]]]), np.zeros(1))
        x, u = np.array([[.1], [.7]]), np.array([[.4], [-.2]])
        g = l.backward(x, u)
        self.assertEqual(g.centers.shape, (4,))
        self.assertEqual(g.log_widths.shape, (4,))
        for key in ['centers', 'log_widths']:
            for k in range(4):
                values = []
                for sign in [1, -1]:
                    b = l.basis
                    a = list(getattr(b, key)); a[k] += sign * 1e-6
                    setattr(b, key, a)
                    q = kan.Layer(1, 1, b); q.set_parameters(l.coefficients, l.bias)
                    values.append(float((q.forward(x) * u).sum()))
                self.assertAlmostEqual(getattr(g, key)[k], (values[0]-values[1])/2e-6, places=8)
        before = np.array(l.basis.centers)
        g.centers[:] = 100  # returned gradient arrays own snapshots
        l.sgd(g, .01)
        self.assertFalse(np.array_equal(l.basis.centers, before))
        self.assertLess(np.max(np.abs(np.array(l.basis.centers)-before)), 1)

    def test_refine_regularize_and_holdout_fit(self):
        l = kan.Layer(1, 1, config(kan.BasisKind.BSpline))
        x = np.linspace(0, 1, 48, dtype=np.float64).reshape(-1, 1)
        target = x*x - .3*x + .2
        for _ in range(400):
            g = l.backward(x, 2*(l.forward(x)-target)/len(x)); l.sgd(g, .3)
        hold = np.array([[.013], [.177], [.411], [.693], [.967]])
        self.assertLess(float(np.mean((l.forward(hold)-(hold*hold-.3*hold+.2))**2)), 1e-6)
        before = l.forward(hold)
        old_g = l.backward(x, np.ones_like(x))
        l.insert_knot(.4); l.adapt_grid(np.array([.1, .15, .2, .8]))
        np.testing.assert_allclose(l.forward(hold), before, rtol=1e-13, atol=1e-13)
        with self.assertRaises(ValueError): l.sgd(old_g, .1)
        value, g = l.regularization(.2)
        self.assertAlmostEqual(value, float(.1*(l.coefficients**2).sum()))
        np.testing.assert_allclose(g.coefficients, .2*l.coefficients)
        with self.assertRaises(TypeError): l.adapt_grid(np.array([.1], dtype=np.float32))
        n = kan.Network([l]); n.insert_knot(0, .6)
        np.testing.assert_allclose(n.forward(hold), before, atol=1e-13)

    @unittest.skipUnless('--cuda' in sys.argv, 'CPU-only build')
    def test_resident_new_families(self):
        self.assertTrue(kan.cuda_available())
        for kind in [kan.BasisKind.BSpline, kan.BasisKind.MexicanHat, kan.BasisKind.GaussianRbf]:
            l = kan.Layer(1, 1, config(kind))
            l.set_parameters(np.array([[[.2, -.3, .4, .1]]]), np.array([.02]))
            cpu = kan.Network([l]); gpu = kan.ResidentNetwork(cpu, 3)
            x, u = np.array([[.1], [.4], [.8]]), np.ones((3, 1))
            gpu.upload_input(x); gpu.upload_output_gradient(u)
            allocations = gpu.workspace_allocations
            for _ in range(2):
                gpu.forward(); np.testing.assert_allclose(gpu.download_output(), cpu.forward(x), atol=1e-12)
                gpu.backward(.2)
                _, rg = cpu.regularization(.2)
                cg = cpu.backward(x, u); gg = gpu.download_gradients()
                np.testing.assert_allclose(gg.layers[0].coefficients, cg.layers[0].coefficients+rg.layers[0].coefficients, atol=1e-12)
                np.testing.assert_allclose(gg.layers[0].centers, cg.layers[0].centers, atol=1e-12)
                gpu.sgd(.01)
                cpu = gpu.download_parameters()
            self.assertEqual(gpu.workspace_allocations, allocations)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
