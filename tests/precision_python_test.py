"""Backlog C1: resident precision policy through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def network():
    first = kan.Layer(3, 4, kan.ChebyshevConfig(5))
    first.set_parameters(0.2 * np.sin(0.7 * np.arange(60)).reshape(4, 3, 5), np.full(4, 0.05))
    second = kan.Layer(4, 2, kan.BSplineConfig(2, [-3, -3, -3, 0, 3, 3, 3]))
    second.set_parameters(0.3 * np.cos(0.4 * np.arange(32)).reshape(2, 4, 4), np.zeros(2))
    return kan.Network([first, kan.InputMap(4, kan.LayerNormMap(epsilon=1e-3, gain=[1.0] * 4, bias=[0.0] * 4)), second])


class Precision(unittest.TestCase):
    def test_enum(self):
        self.assertEqual({p.name for p in kan.Precision}, {"FLOAT64", "FLOAT32"})

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_default_is_float64(self):
        self.assertEqual(kan.ResidentNetwork(network(), 4).precision, kan.Precision.FLOAT64)
        gpu = kan.ResidentNetwork(network(), 4, precision=kan.Precision.FLOAT32)
        self.assertEqual(gpu.precision, kan.Precision.FLOAT32)

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_float32_trains_like_fp64_cpu(self):
        cpu = network()
        x = 0.9 * np.sin(np.arange(15, dtype=float)).reshape(5, 3)
        u = np.cos(np.arange(10, dtype=float)).reshape(5, 2)
        gpu = kan.ResidentNetwork(cpu, 5, precision=kan.Precision.FLOAT32)
        gpu.upload_input(x)
        gpu.upload_output_gradient(u)
        for _ in range(3):
            gpu.forward()
            output = gpu.download_output()
            self.assertEqual(output.dtype, np.float64)
            np.testing.assert_allclose(output, cpu.forward(x), rtol=2e-4, atol=2e-5)
            gpu.backward()
            gg, cg = gpu.download_gradients(), cpu.backward(x, u)
            np.testing.assert_allclose(gg.input, cg.input, rtol=2e-4, atol=2e-5)
            for a, b in zip(gg.layers, cg.layers):
                if isinstance(a, kan.LayerGradients):
                    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=2e-4, atol=2e-5)
            gpu.sgd(0.05)
            cpu.sgd(cg, 0.05)
        for a, b in zip(gpu.download_parameters().layers, cpu.layers):
            if isinstance(b, kan.Layer):
                np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=2e-4, atol=2e-5)

    @unittest.skipUnless(CUDA, "CUDA build")
    def test_float32_keeps_strict_float64_arrays(self):
        gpu = kan.ResidentNetwork(network(), 2, precision=kan.Precision.FLOAT32)
        with self.assertRaises(TypeError):
            gpu.upload_input(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            gpu.upload_input(np.full((1, 3), 1e39))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
