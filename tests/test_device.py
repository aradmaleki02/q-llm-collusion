import io
import tempfile
import unittest
from pathlib import Path

import torch

from collusion.config import Config
from collusion.engine import make_experiment, run_experiment
from collusion.learner import Learner, Network
from collusion.market import outcome
from collusion.storage import Store


class DeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_padding_preserves_variable_length_outputs_and_gradients(self):
        torch.manual_seed(17)
        cpu = Network(Config(hidden_size=32, head_hidden_size=128))
        gpu = Network(Config(hidden_size=32, head_hidden_size=128)).to('cuda:0')
        gpu.load_state_dict(cpu.state_dict())
        sequences = [torch.randn(n, 7) for n in (1, 7, 33)]
        a, b = cpu(sequences), gpu(sequences)
        # cuDNN may use TF32; tolerate its small FP32-reference rounding error.
        torch.testing.assert_close(a, b.cpu(), rtol=1e-3, atol=1e-4)
        a.square().mean().backward()
        b.square().mean().backward()
        for x, y in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=2e-3, atol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_optimizer_checkpoint_can_resume_after_cpu_load(self):
        cfg = Config(device='cuda:0', warmup_transitions=1, hidden_size=8)
        learner = Learner(cfg, 1., 1)
        prices = [1.8]*4
        demand, profit = outcome(prices, cfg)
        learner.learn('test', [dict(prices=prices, demand=demand, profit=profit)], 20)
        buffer = io.BytesIO()
        torch.save(learner, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu', weights_only=False).to('cuda:0')
        restored._update()
        self.assertEqual(restored.updates, 2)
        self.assertEqual(next(restored.online.parameters()).device.type, 'cuda')

    def test_cooperative_pause_blocks_calls_before_starting_next_round(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_experiment(store, 'pause', Config(rounds=2), 'mock', ['ordinary'], [1], 1, 1)
                Path(directory, 'pause_requested').touch()
                result = run_experiment(store, 'pause', progress=lambda _: None)
                self.assertIn('Pause requested', result)
                self.assertEqual(store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 0)
            finally:
                store.close()
