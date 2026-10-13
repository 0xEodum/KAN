"""Backlog R9: ResidentNetwork.upload_parameters through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def network(gain=1.0):
    first = kan.Layer(3, 4, kan.ChebyshevConfig(5))
    first.set_parameters(0.2 * np.sin(0.7 * np.arange(60)).reshape(4, 3, 5), np.full(4, 0.05))
    rbf = kan.Layer(4, 3, kan.TrainableRbfConfig([-1.0, 0.0, 1.0], [-0.2, -0.1, -0.2]))
    rbf.set_parameters(0.3 * np.cos(0.4 * np.arange(36)).reshape(3, 4, 3), np.zeros(3))
    norm = kan.InputMap(4, kan.LayerNormMap(epsilon=1e-3, gain=[gain] * 4, bias=[0.0] * 4))
    return kan.Network([first, norm, rbf])


def trained(model, x, u, steps=3, rate=0.05):
    for _ in range(steps):
        model.sgd(model.backward(x, u), rate)
    return model


def coefficients(model):
    return [layer.coefficients for layer in model.layers if isinstance(layer, kan.Layer)]


@unittest.skipUnless(CUDA, "CUDA build")
class UploadParameters(unittest.TestCase):
    x = 0.9 * np.sin(np.arange(15, dtype=float)).reshape(5, 3)
    u = np.cos(np.arange(15, dtype=float)).reshape(5, 3)

    def check_matches_cpu(self, gpu, cpu, rtol, atol):
        gpu.upload_input(self.x)
        gpu.upload_output_gradient(self.u)
        gpu.forward()
        np.testing.assert_allclose(gpu.download_output(), cpu.forward(self.x), rtol=rtol, atol=atol)
        gpu.backward()
        gg, cg = gpu.download_gradients(), cpu.backward(self.x, self.u)
        np.testing.assert_allclose(gg.input, cg.input, rtol=rtol, atol=atol)
        for a, b in zip(gg.layers, cg.layers):
            if isinstance(a, kan.LayerGradients):
                np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=rtol, atol=atol)
                np.testing.assert_allclose(a.centers, b.centers, rtol=rtol, atol=atol)
                np.testing.assert_allclose(a.log_widths, b.log_widths, rtol=rtol, atol=atol)
            else:
                np.testing.assert_allclose(a.gain, b.gain, rtol=rtol, atol=atol)

    def test_cpu_trained_weights_run_on_the_gpu(self):
        for precision, rtol, atol in ((kan.Precision.FLOAT64, 1e-11, 1e-12), (kan.Precision.FLOAT32, 2e-4, 2e-5)):
            gpu = kan.ResidentNetwork(network(), 5, precision=precision)
            cpu = trained(network(), self.x, self.u)
            self.assertIsNone(gpu.upload_parameters(cpu))
            for a, b in zip(coefficients(gpu.download_parameters()), coefficients(cpu)):
                np.testing.assert_allclose(a, b, rtol=1e-7 if precision == kan.Precision.FLOAT32 else 0, atol=0)
            self.check_matches_cpu(gpu, cpu, rtol, atol)

    def test_round_trip(self):
        gpu = kan.ResidentNetwork(network(), 5)
        gpu.upload_input(self.x)
        gpu.upload_output_gradient(self.u)
        for _ in range(2):
            gpu.forward(); gpu.backward(); gpu.sgd(0.05)
        saved = gpu.download_parameters()
        fresh = kan.ResidentNetwork(network(), 5)
        fresh.upload_parameters(saved)
        for a, b in zip(coefficients(fresh.download_parameters()), coefficients(saved)):
            np.testing.assert_array_equal(a, b)

    def test_mismatch_raises_value_error_and_keeps_state(self):
        gpu = kan.ResidentNetwork(network(), 5)
        gpu.upload_input(self.x)
        gpu.forward()
        output = gpu.download_output()
        other = kan.Network([kan.Layer(3, 4, kan.LegendreConfig(5)), kan.InputMap(4, kan.LayerNormMap(epsilon=1e-3, gain=[1.0] * 4, bias=[0.0] * 4)),
                             kan.Layer(4, 3, kan.TrainableRbfConfig([-1.0, 0.0, 1.0], [-0.2, -0.1, -0.2]))])
        with self.assertRaises(ValueError):
            gpu.upload_parameters(other)
        with self.assertRaises(ValueError):
            gpu.upload_parameters(kan.Network([kan.Layer(3, 2, kan.ChebyshevConfig(5))]))
        np.testing.assert_array_equal(gpu.download_output(), output)

    def test_float32_rejects_unrepresentable_values(self):
        gpu = kan.ResidentNetwork(network(), 5, precision=kan.Precision.FLOAT32)
        before = coefficients(gpu.download_parameters())
        with self.assertRaises(ValueError):
            gpu.upload_parameters(network(gain=1e39))
        for a, b in zip(coefficients(gpu.download_parameters()), before):
            np.testing.assert_array_equal(a, b)
        kan.ResidentNetwork(network(), 5).upload_parameters(network(gain=1e39))  # FP64 accepts

    def test_upload_invalidates_forward_state(self):
        gpu = kan.ResidentNetwork(network(), 5)
        gpu.upload_input(self.x)
        gpu.upload_output_gradient(self.u)
        gpu.forward()
        gpu.upload_parameters(trained(network(), self.x, self.u))
        with self.assertRaises(RuntimeError):
            gpu.backward()
        with self.assertRaises(RuntimeError):
            gpu.download_output()
        gpu.forward()  # input and upstream are kept
        gpu.backward()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
