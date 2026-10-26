import tempfile
import unittest

import torch

from collusion.config import Config
from collusion.convergence import advance_training, plateau
from collusion.engine import make_experiment, run_experiment
from collusion.storage import Store


def policy(**updates):
    value = dict(authorization='Test adaptive training', minimum_games=2, return_window_games=1, return_relative_tolerance=1.,
                 policy_change_tolerance=1., q_drift_tolerance=1., required_stable_checks=1,
                 probe_source_games=1, probe_prefixes_per_game=2,
                 extra_replay_updates_per_game=2, evaluation_reserve_usd=0.)
    value.update(updates)
    return value


class ConvergenceTests(unittest.TestCase):
    def test_plateau_requires_enough_games_and_stable_policy_as_well_as_returns(self):
        rules = policy(minimum_games=6, return_window_games=3, return_relative_tolerance=.05,
                       policy_change_tolerance=.02, q_drift_tolerance=.02)
        rows = [dict(total_profit=100, discounted_profit=2, policy_change_fraction=.01,
                     q_relative_drift=.01) for _ in range(6)]
        self.assertFalse(plateau(rows[:5], rules))
        self.assertTrue(plateau(rows, rules))
        rows[-1]['policy_change_fraction'] = .1
        self.assertFalse(plateau(rows, rules))
        rows[-1]['policy_change_fraction'] = .01
        rows[-1]['discounted_profit'] = 4
        self.assertFalse(plateau(rows, rules))

    def test_repeated_games_keep_learning_then_evaluation_is_frozen(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                cfg = Config(rounds=3, hidden_size=8, warmup_transitions=1, batch_size=2)
                make_experiment(store, 'adaptive', cfg, 'mock', ['ordinary', 'optimistic'], [1], 1, 1, False)
                rules = policy()
                updates_before_eval = None
                for _ in range(10):
                    summary = advance_training(store, 'adaptive', rules)
                    _, state, _ = store.load('adaptive')
                    if summary['stop_reason']:
                        updates_before_eval = updates_before_eval or {
                            k: v.updates for k, v in state['learners'].items()}
                    result = run_experiment(store, 'adaptive', progress=lambda _: None, stop_after_task=True)
                    if result == 'complete':
                        break
                else:
                    self.fail('Adaptive test did not complete')
                _, state, status = store.load('adaptive')
                self.assertEqual(status, 'complete')
                self.assertEqual(state['training_stop_reason'], 'operational_convergence')
                self.assertEqual([t['phase'] for t in state['tasks']], ['train']*4+['eval']*2)
                for key, learner in state['learners'].items():
                    self.assertEqual(learner.steps, 6)
                    self.assertEqual(learner.updates, updates_before_eval[key])
                self.assertEqual(store.db.execute('SELECT count(*) FROM requests').fetchone()[0], 6*3*3)
                # Reopening does not repeat local optimization or duplicate a game.
                before = {k:v.updates for k,v in state['learners'].items()}
                advance_training(store, 'adaptive', rules)
                _, state, _ = store.load('adaptive')
                self.assertEqual(before, {k:v.updates for k,v in state['learners'].items()})
            finally:
                store.close()

    def test_budget_stop_is_not_labeled_convergence(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                cfg = Config(rounds=2, hidden_size=8, warmup_transitions=1)
                make_experiment(store, 'budgeted', cfg, 'mock', ['ordinary', 'optimistic'], [1], 1, 1, False)
                run_experiment(store, 'budgeted', progress=lambda _: None, stop_after_task=True)
                run_experiment(store, 'budgeted', progress=lambda _: None, stop_after_task=True)
                summary = advance_training(store, 'budgeted', policy(evaluation_reserve_usd=51))
                self.assertEqual(summary['stop_reason'], 'budget_limited_not_converged')
                _, state, _ = store.load('budgeted')
                self.assertEqual(state['tasks'][state['task_index']]['phase'], 'eval')
            finally:
                store.close()
