from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from unittest.mock import patch

import torch

from collusion.api import Gateway
from collusion.config import Config
from collusion.engine import make_experiment, run_experiment
from collusion.storage import Paused, Store


class EfficiencyTests(unittest.TestCase):
    def test_more_replay_updates_require_no_extra_opponent_calls(self):
        torch.set_num_threads(1)
        config = Config(rounds=8, hidden_size=8, warmup_transitions=1, updates_per_step=4,
                        experimental_seller=3, prompt_cache_blocks=True, action_interface='price')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_experiment(store, 'efficient', config, 'mock', ['ordinary'], [1], 1, 1, False)
                run_experiment(store, 'efficient', progress=lambda _: None)
                _, state, _ = store.load('efficient')
                learner = state['learners']['ordinary:1']
                self.assertEqual(learner.steps, 8)
                self.assertEqual(learner.updates, 32)
                self.assertEqual(store.db.execute('SELECT COUNT(*) FROM requests').fetchone()[0], 16*3)
            finally:
                store.close()

    def test_parallel_reservations_respect_stage_limit_without_changing_shared_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Store(directory)
            initial.close()
            def reserve(index):
                store = Store(directory)
                try:
                    try:
                        store.reserve('stage', str(index), {}, .03, spend_ceiling=.05)
                        return True
                    except Paused:
                        return False
                finally:
                    store.close()
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(pool.map(reserve, range(2))), 1)
            store = Store(directory)
            try:
                self.assertEqual(store.budget(), 50)
                self.assertAlmostEqual(store.spending()['reserved'], .03)
            finally:
                store.close()

    def test_engine_applies_saved_stage_limit_before_network_calls(self):
        torch.set_num_threads(1)
        config = Config(rounds=2, hidden_size=8, parallel_llm_calls=4)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                state = make_experiment(store, 'limited', config, 'openai', ['ordinary'], [1], 1, 1, False)
                state['spend_ceiling'] = 0
                store.save('limited', config, state)
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'not-a-real-key'}), \
                     patch.object(Gateway, '_send') as send:
                    result = run_experiment(store, 'limited', progress=lambda _: None)
                self.assertTrue(result.startswith('paused'))
                self.assertIn('Stage spending limit', result)
                send.assert_not_called()
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
