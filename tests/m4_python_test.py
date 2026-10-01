"""Independent rational identities, strict interfaces and nonlinear learning."""
import sys
import unittest
import numpy as np
import kan


def rational(m=1, n=1):
    c = kan.RationalConfig()
    c.numerator_degree, c.denominator_degree = m, n
    return c


def layer():
    l = kan.Layer(1, 1, rational())
    l.set_rational_parameters(np.array([[[1., .5]]]), np.array([[[-.5]]]), np.zeros(1))
    return l


class M4(unittest.TestCase):
    def test_pade_and_owned_snapshots(self):
        l = layer()
        x = np.array([[-.4], [0.], [.7]])
        np.testing.assert_allclose(l.forward(x), (1+.5*x)/(1-.5*x), atol=1e-14)
        self.assertTrue(l.is_rational)
        with self.assertRaises(ValueError): _ = l.basis
        c = l.rational_config; c.center = 99
        self.assertEqual(l.rational_config.center, 0)
        l.denominators[:] = 99
        np.testing.assert_array_equal(l.denominators, [[[-.5]]])
        g = l.backward(x, np.ones_like(x))
        v, dx, da, db = kan.evaluate_rational(rational(), .5, np.array([1.,.5]), np.array([-.5]))
        self.assertAlmostEqual(v, 5/3)
        self.assertAlmostEqual(dx, 16/9)
        np.testing.assert_allclose(da,[4/3,2/3])
        np.testing.assert_allclose(db,[-10/9])
        np.testing.assert_allclose(g.input, 1/(1-.5*x)**2, atol=1e-14)
        self.assertEqual(g.denominators.shape, (1, 1, 1))
        g.denominators[:] = 99
        before = l.denominators.copy()
        l.sgd(g, .001)
        self.assertLess(float(np.max(np.abs(l.denominators-before))), .01)

    def test_parameter_vjps_and_strict_validation(self):
        l = layer(); x = np.array([[-.6], [.1], [.8]]); u = np.array([[.2], [-.3], [.4]])
        g = l.backward(x, u)
        for key in ['coefficients', 'denominators', 'bias']:
            initial = getattr(l, key)
            for index in np.ndindex(initial.shape):
                values = []
                for sign in [1, -1]:
                    a, b, bias = l.coefficients, l.denominators, l.bias
                    target = {'coefficients': a, 'denominators': b, 'bias': bias}[key]
                    target[index] += sign * 1e-6
                    q = layer(); q.set_rational_parameters(a, b, bias)
                    values.append(float((q.forward(x)*u).sum()))
                self.assertAlmostEqual(getattr(g, key)[index], (values[0]-values[1])/2e-6, places=8)
        with self.assertRaises(TypeError): l.forward(x.astype(np.float32))
        with self.assertRaises(ValueError): l.set_rational_parameters(l.coefficients, np.zeros((1,1,2)), l.bias)
        with self.assertRaises(ValueError): l.set_parameters(l.coefficients, l.bias)
        with self.assertRaises(ValueError): l.forward(np.array([[2.]]))
        self.assertEqual(l.forward(np.empty((0, 1))).shape, (0, 1))
        z = kan.Layer(1, 1, rational(0, 0))
        z.set_rational_parameters(np.array([[[.3]]]), np.empty((1,1,0)), np.zeros(1))
        self.assertEqual(z.backward(x, u).denominators.shape, (1,1,0))
        b = kan.Layer(1,1,kan.ChebyshevConfig())
        self.assertEqual(b.denominators.shape, (0,))

    def test_learning_with_independent_holdout_and_mixed_network(self):
        c = rational(); c.center, c.scale = .2, 1.5
        l = kan.Layer(1, 1, c)
        l.set_rational_parameters(np.array([[[.2, .1]]]), np.array([[[.05]]]), np.zeros(1))
        x = np.linspace(-1,1,64).reshape(-1,1)
        z = (x-.2)/1.5; target = (.3+.8*z)/(1+.4*z)
        for _ in range(1400):
            l.sgd(l.backward(x, 2*(l.forward(x)-target)/len(x)), .15)
        hold = np.array([[-.977], [-.351], [.073], [.527], [.913]])
        h = (hold-.2)/1.5
        self.assertLess(float(np.mean((l.forward(hold)-(.3+.8*h)/(1+.4*h))**2)), 1e-6)
        self.assertGreater(float(np.abs(l.denominators-.05).max()), .05)
        basis = kan.ChebyshevConfig(size=2)
        b = kan.Layer(1,1,basis); b.set_parameters(np.array([[[.1,.7]]]), np.array([.02]))
        n = kan.Network([l,b]); ng = n.backward(hold,np.ones_like(hold)); n.sgd(ng,.001)
        self.assertEqual(ng.layers[0].denominators.shape,(1,1,1))
        self.assertEqual(ng.layers[1].denominators.shape,(0,))
        penalty, pg = l.regularization(.2)
        self.assertAlmostEqual(penalty, float(.1*(l.coefficients*l.coefficients).sum()))
        np.testing.assert_array_equal(pg.denominators, np.zeros((1,1,1)))

    @unittest.skipUnless('--cuda' in sys.argv, 'CPU-only build')
    def test_resident_and_singularity_recovery(self):
        cpu = kan.Network([layer()]); gpu = kan.ResidentNetwork(cpu,3)
        x = np.array([[-.4],[0.],[.7]]); u = np.array([[.2],[-.1],[.3]])
        allocations = gpu.workspace_allocations
        gpu.upload_input(x); gpu.upload_output_gradient(u)
        for _ in range(3):
            gpu.forward(); np.testing.assert_allclose(gpu.download_output(),cpu.forward(x),atol=1e-12)
            gpu.backward(); gg = gpu.download_gradients(); cg = cpu.backward(x,u)
            for key in ['input','coefficients','denominators','bias']:
                np.testing.assert_allclose(getattr(gg.layers[0],key),getattr(cg.layers[0],key),atol=1e-12)
            gpu.sgd(.001); cpu.sgd(cg,.001)
            np.testing.assert_allclose(gpu.download_parameters().layers[0].denominators,cpu.layers[0].denominators,atol=1e-12)
        self.assertEqual(allocations,gpu.workspace_allocations)
        pole = kan.ResidentNetwork(kan.Network([layer()]),3)
        pole.upload_input(np.array([[2.]])); pole.upload_output_gradient(np.ones((1,1)))
        with self.assertRaises(ValueError): pole.forward()
        with self.assertRaises(RuntimeError): pole.download_output()
        pole.upload_input(x); pole.upload_output_gradient(u); pole.forward(); pole.backward()
        np.testing.assert_allclose(pole.download_output(),layer().forward(x),atol=1e-12)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
