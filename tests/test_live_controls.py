import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from collusion.api import Gateway, parse_decision, prompt_body
from collusion.config import Config
from collusion.engine import make_baseline, make_experiment, run_experiment
from collusion.learner import observations
from collusion.storage import Paused, Store


class LiveControlTests(unittest.TestCase):
    def test_longer_planning_changes_only_the_private_note_allowance(self):
        short = Config(action_interface='price', strategy_words=40)
        long = replace(short, strategy_words=160)
        a, b = prompt_body(short, 3, [], ''), prompt_body(long, 3, [], '')
        a['input'][0]['content'] = a['input'][0]['content'].replace('at most 40 words', 'at most 160 words')
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            replace(short, strategy_words=0).validate()
    def test_extend_unannounced_baseline_preserves_history_and_objective(self):
        from collusion.engine import make_baseline
        torch.set_num_threads(1)
        config = Config(rounds=3, action_interface='price')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_baseline(store, 'baseline', config, backend='mock')
                run_experiment(store, 'baseline', progress=lambda _: None, step_limit=2)
                before = [dict(row) for row in store.records('baseline')]
                _, state, _ = store.load('baseline')
                original_body = prompt_body(config, 3, state['market']['history'], '')
                store.extend_baseline_horizon('baseline', 5)
                amended, state, _ = store.load('baseline')
                self.assertEqual(prompt_body(Config(**amended), 3, state['market']['history'], ''), original_body)
                self.assertEqual(state['horizon_amendments'][0]['after_committed_round'], 2)
                run_experiment(store, 'baseline', progress=lambda _: None)
                after = [dict(row) for row in store.records('baseline')]
                self.assertEqual(len(after), 5)
                self.assertEqual(after[:2], before)
            finally:
                store.close()

    def test_finite_game_horizon_cannot_be_rewritten(self):
        from collusion.engine import make_baseline
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_baseline(store, 'finite', Config(rounds=3, gamma=1), backend='mock')
                with self.assertRaises(ValueError):
                    store.extend_baseline_horizon('finite', 5)
            finally:
                store.close()

    def test_actual_price_interface_removes_index_conversion(self):
        config = Config(action_interface='price')
        body = prompt_body(config, 3, [], '')
        self.assertIn('price', body['text']['format']['schema']['properties'])
        self.assertNotIn('price_index', body['text']['format']['schema']['properties'])
        def response(value):
            return {'status': 'completed', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps({'price': value, 'strategy': 'test'})}]}]}
        self.assertEqual(parse_decision(response(1.20), config)['index'], 5)
        for value in (1.21, float('nan'), True, '1.20'):
            with self.assertRaises(ValueError):
                parse_decision(response(value), config)

    def test_fourth_seller_observations_and_training(self):
        torch.set_num_threads(1)
        config = Config(rounds=3, experimental_seller=3, parallel_llm_calls=4,
                        warmup_transitions=1, hidden_size=8, action_interface='price')
        row = dict(prices=[1.2, 1.4, 1.6, 1.8], demand=[.1, .2, .3, .4], profit=[1., 2., 3., 4.])
        self.assertEqual(observations([row], config)[-1, 5].item(), 4.)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_experiment(store, 'fourth', config, 'mock', ['optimistic'], [1], 1, 1)
                self.assertEqual(run_experiment(store, 'fourth', progress=lambda _: None), 'complete')
                _, state, _ = store.load('fourth')
                self.assertEqual(state['learners']['optimistic:1'].steps, 3)
                requests = list(store.db.execute("SELECT position FROM requests WHERE position LIKE '0/%'"))
                self.assertEqual(len(requests), 9)
                self.assertTrue(all(not r['position'].endswith('/3') for r in requests))
            finally:
                store.close()

    def test_simultaneous_reservations_cannot_overspend(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Store(directory)
            initial.budget(.1)
            initial.close()
            barrier = threading.Barrier(4)

            def reserve(index):
                store = Store(directory)
                try:
                    barrier.wait(timeout=10)
                    try:
                        store.reserve('parallel', str(index), {}, .06)
                        return True
                    except Paused:
                        return False
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=4) as pool:
                successes = list(pool.map(reserve, range(4)))
            self.assertEqual(sum(successes), 1)
            store = Store(directory)
            try:
                self.assertAlmostEqual(store.spending()['reserved'], .06)
            finally:
                store.close()

    def test_parallel_baseline_has_identical_information(self):
        torch.set_num_threads(1)
        config = Config(rounds=1, parallel_llm_calls=4)
        barrier = threading.Barrier(4)
        payloads = []

        def send(gateway, body):
            data = json.loads(body['input'][1]['content'])
            payloads.append(data)
            barrier.wait(timeout=10)
            return {'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 10},
                    'output': [{'type': 'message', 'content': [{'type': 'output_text',
                        'text': json.dumps({'price_index': 25, 'strategy': 'test'})}]}]}

        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_baseline(store, 'baseline', config)
                with patch.dict('os.environ', {'OPENAI_API_KEY': 'not-a-real-key'}), \
                     patch.object(Gateway, '_send', send):
                    self.assertEqual(run_experiment(store, 'baseline', progress=lambda _: None), 'complete')
                self.assertEqual(len(payloads), 4)
                self.assertTrue(all(p['complete_market_history'] == [] for p in payloads))
                self.assertEqual({p['your_seller'] for p in payloads}, set('ABCD'))
                self.assertEqual(len(store.records('baseline')), 1)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
