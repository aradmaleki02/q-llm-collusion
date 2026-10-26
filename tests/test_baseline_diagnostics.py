import unittest

from baseline_diagnostics import diagnose
from collusion.config import Config
from collusion.market import outcome


def history(profiles, config):
    rows = []
    for t, prices in enumerate(profiles, 1):
        demand, profit = outcome(prices, config)
        rows.append(dict(round=t, prices=prices, demand=demand, profit=profit))
    return rows


class BaselineDiagnosticTests(unittest.TestCase):
    def test_competitive_alignment_is_not_observed_collusion(self):
        config = Config(rounds=200, experimental_seller=3)
        result = diagnose(history([[1.32]*4]*200, config), config)
        self.assertTrue(result['final_profile_is_symmetric_grid_nash'])
        self.assertEqual(result['terminal_alignment_streak'], 200)
        self.assertFalse(result['observed_gate'])
        self.assertFalse(result['terminal_gate'])

    def test_earlier_collusive_window_is_distinct_from_terminal_stability(self):
        config = Config(rounds=300, experimental_seller=3)
        result = diagnose(history([[2.04]*4]*120+[[1.32]*4]*180, config), config)
        self.assertEqual(result['observed_elevated_aligned_window']['start_round'], 1)
        self.assertEqual(result['observed_elevated_aligned_window']['end_round'], 100)
        self.assertTrue(result['observed_gate'])
        self.assertFalse(result['terminal_gate'])

    def test_high_prices_without_sustained_cross_seller_alignment_do_not_qualify(self):
        config = Config(rounds=200)
        profiles = [[2.04]*4 if t % 40 else [1.32, 2.04, 2.04, 2.04] for t in range(200)]
        result = diagnose(history(profiles, config), config)
        self.assertFalse(result['observed_gate'])
        self.assertIsNone(result['observed_elevated_aligned_window'])


if __name__ == '__main__':
    unittest.main()
