from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from capacity_audit import teacher, successor
from collusion.config import Config
from collusion.learner import Learner
from collusion.market import outcome
from collusion.storage import Paused
from study_controller import warm_start


class LocalCapacityTests(unittest.TestCase):
    def test_reviewed_initialization_preserves_weights_and_resets_live_training(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Config(hidden_size=8, epsilon_start=1.)
            live = replace(local, epsilon_start=.2, epsilon_decay_steps=1000)
            learner = Learner(local, 1., 1)
            learner.steps, learner.updates = 10000, 39748
            original = next(learner.online.parameters()).detach().clone()
            path = Path(directory)/'ordinary.pt'
            torch.save(dict(learner=learner, report={'ready': False, 'results': []}), path)
            policies, provenance = warm_start(live, {'ordinary': str(path)}, require_ready=False)
            result = policies['ordinary:1']
            torch.testing.assert_close(next(result.online.parameters()), original)
            self.assertEqual((result.steps, result.updates, result.replay), (0, 0, []))
            self.assertEqual(result.config, live)
            self.assertEqual(provenance['ordinary']['local_steps'], 10000)
            self.assertFalse(provenance['ordinary']['local_controls_passed'])

    def test_diagnostic_oracle_satisfies_discounted_bellman_equation(self):
        cfg = Config(experimental_seller=3)
        q = teacher(cfg, 1.92)
        for state in range(5):
            for index, price in enumerate(cfg.prices):
                rival = 1.92 if state < 2 else 1.32
                reward = outcome([rival]*3+[price], cfg)[1][3]
                target = reward+cfg.gamma*float(q[successor(state, price, 1.92)].max())
                self.assertAlmostEqual(float(q[state, index]), target, delta=1e-5)

    def test_failed_local_policy_blocks_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(hidden_size=8)
            learner = Learner(cfg, 1., 1)
            path = Path(directory)/'ordinary.pt'
            torch.save(dict(learner=learner, report={'ready': False}), path)
            with self.assertRaises(Paused):
                warm_start(cfg, {'ordinary': str(path)})

    def test_warm_start_checks_architecture_and_loss_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(hidden_size=8)
            learner = Learner(cfg, .25, 1)
            path = Path(directory)/'optimistic.pt'
            torch.save(dict(learner=learner, report={'ready': True, 'results': []}), path)
            with self.assertRaises(Paused):
                warm_start(cfg, {'ordinary': str(path)})
            with self.assertRaises(Paused):
                warm_start(replace(cfg, hidden_size=32), {'optimistic': str(path)})
