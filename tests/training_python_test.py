"""Backlog C9: on-device training steps through the NumPy bindings."""
import sys
import unittest
import numpy as np
import kan

CUDA = "--cuda" in sys.argv


def network():
    first = kan.Layer(3, 4, kan.ChebyshevConfig(5))
    first.set_parameters(0.2 * np.sin(0.7 * np.arange(60)).reshape(4, 3, 5), np.full(4, 0.05))
    second = kan.Layer(4, 2, kan.LegendreConfig(4))
    second.set_parameters(0.3 * np.cos(0.4 * np.arange(32)).reshape(2, 4, 4), np.zeros(2))
    return kan.Network([first, second])


def coefficients(model):
    return np.concatenate([layer.coefficients.ravel() for layer in model.layers])


@unittest.skipUnless(CUDA, "CUDA build")
class Training(unittest.TestCase):
    x = 0.9 * np.sin(np.arange(18, dtype=float)).reshape(6, 3)
    t = 0.5 * np.cos(np.arange(12, dtype=float)).reshape(6, 2)

    def test_loss_enum(self):
        self.assertEqual({kan.Loss.OUTPUT_GRADIENT, kan.Loss.MEAN_SQUARED_ERROR},
                         set(kan.Loss.__members__.values()))

    def test_staged_mse_steps_match_the_cpu(self):
        for precision, rtol in ((kan.Precision.FLOAT64, 1e-10), (kan.Precision.FLOAT32, 2e-3)):
            gpu = kan.ResidentNetwork(network(), 6, precision=precision)
            cpu = network()
            for _ in range(3):
                y = cpu.forward(self.x)
                expected_loss = float(np.mean((y - self.t) ** 2))
                cpu.sgd(cpu.backward(self.x, 2.0 * (y - self.t) / y.size), 0.05)
                self.assertIsNone(gpu.train_step(self.x, self.t, 0.05))
                np.testing.assert_allclose(gpu.download_loss(), expected_loss, rtol=rtol)
            np.testing.assert_allclose(coefficients(gpu.download_parameters()), coefficients(cpu), rtol=rtol, atol=1e-6)
            self.assertEqual(gpu.trained_steps, 3)

    def test_resident_steps(self):
        gpu = kan.ResidentNetwork(network(), 6)
        eager = kan.ResidentNetwork(network(), 6)
        u = np.cos(np.arange(12, dtype=float)).reshape(6, 2)
        for model in (gpu, eager):
            model.upload_input(self.x)
            model.upload_output_gradient(u)
        for _ in range(2):
            gpu.train_step(0.05, coefficient_l2=1e-3)
            eager.forward(); eager.backward(1e-3); eager.sgd(0.05)
        np.testing.assert_array_equal(coefficients(gpu.download_parameters()), coefficients(eager.download_parameters()))
        gpu.upload_target(self.t)
        gpu.train_step(0.05, loss=kan.Loss.MEAN_SQUARED_ERROR)
        self.assertGreater(gpu.download_loss(), 0.0)

    def test_deferred_status(self):
        gpu = kan.ResidentNetwork(network(), 6)
        self.assertEqual(gpu.status_interval, 1)
        gpu.status_interval = 3
        self.assertEqual(gpu.status_interval, 3)
        with self.assertRaises(ValueError):
            gpu.status_interval = 0
        bad = self.x.copy()
        bad[0, 0] = 1e70  # overflows the Chebyshev recurrence in FP64
        gpu.train_step(self.x, self.t, 0.05)
        gpu.train_step(bad, self.t, 0.05)
        with self.assertRaisesRegex(OverflowError, "training step 1"):
            gpu.train_step(self.x, self.t, 0.05)
        self.assertEqual(gpu.trained_steps, 1)
        gpu.check_status()

    def test_strict_arrays(self):
        gpu = kan.ResidentNetwork(network(), 6)
        with self.assertRaises(TypeError):
            gpu.train_step(self.x.astype(np.float32), self.t, 0.05)
        with self.assertRaises(ValueError):
            gpu.train_step(self.x, self.t[:, :1].copy(), 0.05)
        with self.assertRaises(RuntimeError):  # std::logic_error: no target uploaded
            gpu.upload_input(self.x)
            gpu.train_step(0.05, loss=kan.Loss.MEAN_SQUARED_ERROR)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
