import tempfile
import unittest

import torch

from collusion.config import Config
from collusion.engine import make_experiment, run_experiment
from collusion.storage import Store
from study_controller import cap_stage, make_validation, select_candidate


def row(variant, total, discounted, index=50, lift=15):
    return dict(variant=variant, phase='eval', complete='True', focal_total_profit=str(total),
                focal_discounted_profit=str(discounted), profit_collusion_index_pct=str(index),
                price_lift_last100_pct=str(lift), full_run_profit_index_pct=str(index),
                full_run_price_lift_pct=str(lift))


class StudyTests(unittest.TestCase):
    def test_removing_stage_cap_clears_saved_limit_and_keeps_shared_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_experiment(store, 'study', Config(hidden_size=8), 'mock', ['ordinary'], [1], 1, 1)
                cap_stage(store, 'study', .01)
                self.assertIsNone(cap_stage(store, 'study', None))
                _, state, _ = store.load('study')
                self.assertIsNone(state['spend_ceiling'])
                self.assertIsNone(state['stage_allowance_usd'])
                self.assertEqual(state['budget_amendments'][0]['previous_stage_allowance'], .01)
                self.assertEqual(store.budget(), 50)
                store.reserve('study', 'beyond_old_stage_cap', {}, .02, spend_ceiling=state['spend_ceiling'])
                self.assertIsNone(cap_stage(store, 'study', None))
                _, again, _ = store.load('study')
                self.assertEqual(len(again['budget_amendments']), 1)
            finally:
                store.close()

    def test_stage_allowance_does_not_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                config = Config(rounds=2, hidden_size=8)
                make_experiment(store, 'limited', config, 'mock', ['ordinary'], [1], 1, 1)
                self.assertAlmostEqual(cap_stage(store, 'limited', .01), .01)
                call = store.reserve('limited', 'one', {}, .005)
                store.received(call, {}, .005)
                self.assertAlmostEqual(cap_stage(store, 'limited', .02), .01)
                self.assertEqual(store.budget(), 50)
            finally:
                store.close()

    def test_profit_without_collusion_does_not_qualify(self):
        selected, all_results = select_candidate([
            row('ordinary', 40, 1), row('all_llm', 50, 2), row('optimistic', 60, 3, index=-10)])
        self.assertIsNone(selected)
        self.assertEqual(len(all_results), 1)  # Negative result remains visible.

    def test_total_gain_cannot_hide_discounted_objective_loss(self):
        selected, _ = select_candidate([
            row('ordinary', 40, 1), row('all_llm', 50, 2), row('optimistic', 60, 1.5)])
        self.assertIsNone(selected)
        selected, _ = select_candidate([
            row('ordinary', 40, 1), row('all_llm', 50, 2), row('optimistic', 60, 3)])
        self.assertEqual(selected['variant'], 'optimistic')

    def test_development_can_reuse_explicit_baseline_without_extra_paid_reference(self):
        baseline = dict(row('all_llm', 50, 2), reference_source='established_baseline')
        selected, comparisons = select_candidate([
            row('ordinary', 40, 1), row('optimistic', 60, 3)], external_reference=baseline)
        self.assertEqual(selected['variant'], 'optimistic')
        self.assertEqual(comparisons[0]['llm_reference_source'], 'established_baseline')
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                state = make_experiment(store, 'small', Config(rounds=2, hidden_size=8), 'mock',
                                        ['ordinary', 'optimistic'], [1], 1, 1, include_llm_reference=False)
                self.assertEqual(len(state['tasks']), 4)
                self.assertTrue(all(t['variant'] != 'all_llm' for t in state['tasks']))
            finally:
                store.close()

    def test_validation_clones_frozen_policies_and_contains_no_training(self):
        torch.set_num_threads(1)
        config = Config(rounds=2, hidden_size=8, warmup_transitions=1, experimental_seller=3)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                make_experiment(store, 'source', config, 'mock', ['ordinary', 'optimistic'], [1], 1, 1)
                run_experiment(store, 'source', progress=lambda _: None)
                _, source, _ = store.load('source')
                name = make_validation(store, 'source', 'optimistic', config)
                _, validation, _ = store.load(name)
                self.assertEqual(len(validation['tasks']), 3)
                self.assertTrue(all(t['phase'] == 'eval' for t in validation['tasks']))
                for key, learner in validation['learners'].items():
                    self.assertEqual(learner.replay, [])
                    self.assertTrue(all(torch.equal(v, source['learners'][key].online.state_dict()[k])
                                        for k, v in learner.online.state_dict().items()))
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
