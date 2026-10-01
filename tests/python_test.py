"""Independent numerical and strict-array checks for the optional bindings."""
import argparse
import gc
import unittest

import numpy as np
import kan


def config(kind=kan.BasisKind.Chebyshev):
    result = kan.BasisConfig()
    result.kind = kind
    result.size = 5
    result.alpha = 0.3
    result.beta = 0.7
    result.frequency = 1.4
    result.centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    result.width = 0.8
    return result


def layer(inputs, outputs, kind=kan.BasisKind.Chebyshev):
    result = kan.Layer(inputs, outputs, config(kind))
    count = outputs * inputs * 5
    coefficients = np.arange(count, dtype=np.float64).reshape(outputs, inputs, 5)
    result.set_parameters(0.001 * (coefficients - count / 2), np.zeros(outputs))
    return result


class Bindings(unittest.TestCase):
    def test_basis_independent_values_and_derivatives(self):
        x = 0.31
        for kind in [kan.BasisKind.Chebyshev, kan.BasisKind.Legendre,
                     kan.BasisKind.Jacobi, kan.BasisKind.Hermite,
                     kan.BasisKind.Fourier, kan.BasisKind.GaussianRbf]:
            cfg = config(kind)
            values, derivative = kan.evaluate_basis(cfg, x)
            h = 1e-6
            plus, _ = kan.evaluate_basis(cfg, x + h)
            minus, _ = kan.evaluate_basis(cfg, x - h)
            np.testing.assert_allclose(derivative, (plus - minus) / (2*h), atol=2e-9)
            if kind == kan.BasisKind.Chebyshev:
                expected = np.polynomial.chebyshev.chebvander(x, 4).reshape(5)
            elif kind == kan.BasisKind.Legendre:
                expected = np.polynomial.legendre.legvander(x, 4).reshape(5)
            elif kind == kan.BasisKind.Hermite:
                expected = np.polynomial.hermite.hermvander(x, 4).reshape(5)
            elif kind == kan.BasisKind.Fourier:
                expected = [1, np.cos(1.4*x), np.sin(1.4*x), np.cos(2.8*x), np.sin(2.8*x)]
            elif kind == kan.BasisKind.GaussianRbf:
                expected = np.exp(-((x - np.array(cfg.centers)) / cfg.width)**2)
            else:
                # Independently evaluate Jacobi through the generalized binomial sum.
                import math
                def choose(a, k):
                    return math.prod(a-j for j in range(k)) / math.factorial(k)
                expected = [sum(choose(n+cfg.alpha, n-k) * choose(n+cfg.beta, k)
                                * ((x-1)/2)**k * ((x+1)/2)**(n-k)
                                for k in range(n+1)) for n in range(5)]
            np.testing.assert_allclose(values, expected, atol=1e-13)
            self.assertTrue(values.flags.owndata and derivative.flags.owndata)

    def test_layer_gradient_shapes_and_finite_differences(self):
        model = layer(2, 2)
        x = np.array([[-0.4, 0.2], [0.7, -0.1]])
        upstream = np.array([[0.3, -0.5], [0.2, 0.8]])
        gradient = model.backward(x, upstream)
        self.assertEqual(gradient.input.shape, x.shape)
        self.assertEqual(gradient.coefficients.shape, (2, 2, 5))
        self.assertEqual(gradient.bias.shape, (2,))
        h = 1e-6
        coefficients, bias = model.coefficients, model.bias
        for index in np.ndindex(coefficients.shape):
            plus, minus = coefficients.copy(), coefficients.copy()
            plus[index] += h
            minus[index] -= h
            model.set_parameters(plus, bias)
            fp = np.sum(model.forward(x) * upstream)
            model.set_parameters(minus, bias)
            fm = np.sum(model.forward(x) * upstream)
            self.assertAlmostEqual(gradient.coefficients[index], (fp-fm)/(2*h), places=9)
        model.set_parameters(coefficients, bias)
        for index in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[index] += h
            minus[index] -= h
            fd = np.sum((model.forward(plus)-model.forward(minus))*upstream)/(2*h)
            self.assertAlmostEqual(gradient.input[index], fd, places=9)
        np.testing.assert_allclose(gradient.bias, upstream.sum(axis=0))
        model.sgd(gradient, 0.1)
        np.testing.assert_allclose(model.coefficients, coefficients - 0.1*gradient.coefficients)

    def test_network_finite_difference_and_owned_snapshots(self):
        model = kan.Network([layer(2, 3), layer(3, 1, kan.BasisKind.Legendre)])
        x = np.array([[-0.4, 0.2], [0.7, -0.1]])
        upstream = np.array([[0.3], [-0.5]])
        gradient = model.backward(x, upstream, batch=2)
        self.assertEqual(len(gradient.layers), 2)
        h = 1e-6
        for index in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[index] += h
            minus[index] -= h
            fd = np.sum((model.forward(plus)-model.forward(minus))*upstream)/(2*h)
            self.assertAlmostEqual(gradient.input[index], fd, places=9)
        snapshots = model.layers
        snapshots[0].set_parameters(np.zeros((3, 2, 5)), np.ones(3))
        self.assertFalse(np.array_equal(snapshots[0].coefficients, model.layers[0].coefficients))
        output = model.forward(x)
        owned = gradient.layers[0].coefficients
        del gradient, model
        gc.collect()
        self.assertTrue(output.flags.owndata and owned.flags.owndata)
        self.assertTrue(np.isfinite(output).all() and np.isfinite(owned).all())

    def test_strict_validation(self):
        model = layer(2, 2)
        x = np.ones((3, 2))
        for invalid in [x.astype(np.float32), x.astype(np.int64), x[:, ::-1],
                        np.asfortranarray(x), x.ravel(), x.tolist(), np.ones((3, 3)),
                        x.astype('>f8')]:
            with self.assertRaises((TypeError, ValueError)):
                model.forward(invalid)
        with self.assertRaises(ValueError):
            model.forward(x, batch=2)
        with self.assertRaises(ValueError):
            model.backward(x, np.ones((3, 1)))
        with self.assertRaises(ValueError):
            model.set_parameters(np.ones((2, 10)), np.zeros(2))
        bad = x.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.forward(bad)
        with self.assertRaises(ValueError):
            model.sgd(model.backward(x, x), -0.1)
        self.assertEqual(model.forward(np.empty((0, 2))).shape, (0, 2))

    @unittest.skipUnless('--cuda' in __import__('sys').argv, 'CPU-only binding run')
    def test_resident_all_families_training_and_validation(self):
        self.assertTrue(kan.cuda_enabled)
        kinds = [kan.BasisKind.Chebyshev, kan.BasisKind.Legendre, kan.BasisKind.Jacobi,
                 kan.BasisKind.Hermite, kan.BasisKind.Fourier, kan.BasisKind.GaussianRbf]
        x = np.array([[-0.4, 0.2], [0.7, -0.1], [0.1, 0.3]])
        upstream = np.array([[0.3], [-0.5], [0.2]])
        for kind in kinds:
            cpu = kan.Network([layer(2, 3, kind), layer(3, 1, kind)])
            gpu = kan.ResidentNetwork(cpu, 4)
            allocations = gpu.workspace_allocations
            self.assertEqual(gpu.capacity, 4)
            gpu.upload_input(x)
            self.assertEqual(gpu.batch, 3)
            gpu.upload_output_gradient(upstream)
            for _ in range(3):
                gpu.forward()
                np.testing.assert_allclose(gpu.download_output(), cpu.forward(x), atol=1e-12)
                gpu.backward()
                gradients = gpu.download_gradients()
                reference = cpu.backward(x, upstream)
                np.testing.assert_allclose(gradients.input, reference.input, atol=1e-12)
                for actual, expected in zip(gradients.layers, reference.layers):
                    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-12)
                    np.testing.assert_allclose(actual.bias, expected.bias, atol=1e-12)
                cpu.sgd(reference, 0.01)
                gpu.sgd(0.01)
            gpu.synchronize()
            for actual, expected in zip(gpu.download_parameters().layers, cpu.layers):
                np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-12)
                np.testing.assert_allclose(actual.bias, expected.bias, atol=1e-12)
            self.assertEqual(gpu.workspace_allocations, allocations)
            with self.assertRaises(ValueError):
                gpu.upload_input(np.ones((5, 2)))
            with self.assertRaises((TypeError, ValueError)):
                gpu.upload_input(x.astype(np.float32))
            with self.assertRaises(ValueError):
                gpu.upload_output_gradient(np.ones((3, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.parse_args()
    unittest.main(argv=[__import__('sys').argv[0]])
