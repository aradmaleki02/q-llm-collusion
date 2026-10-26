from dataclasses import replace
import json
import unittest

from collusion.api import prompt_body, response_cost
from collusion.config import Config
from collusion.market import outcome


def records(count, config):
    history = []
    for t in range(1, count+1):
        prices = [config.prices[(t+i) % config.price_count] for i in range(4)]
        demand, profit = outcome(prices, config)
        history.append(dict(round=t, prices=prices, demand=demand, profit=profit))
    return history


class PromptCacheTests(unittest.TestCase):
    def test_missing_cache_accounting_uses_conservative_reservation(self):
        config = Config(prompt_cache_blocks=True)
        response = {'usage': {'input_tokens': 10000, 'output_tokens': 100,
                    'input_tokens_details': {'cached_tokens': 9000}}}
        self.assertEqual(response_cost(response, config, .01), .01)
        response['usage']['input_tokens_details']['cache_write_tokens'] = 500
        expected = (500*.10 + 9000*.01 + 500*.125 + 100*.5)/1_000_000
        self.assertAlmostEqual(response_cost(response, config, .01), expected)

    def test_blocks_preserve_every_full_history_field_for_each_seller(self):
        for gamma in (.95, 1.0):
            config = Config(action_interface='price', rounds=1000, gamma=gamma, strategy_words=160)
            for seller in range(4):
                for count in (0, 1, 24, 25, 26, 49, 50, 51, 1000):
                    history = records(count, config)
                    compact = prompt_body(config, seller, history, 'private note')
                    cached = prompt_body(replace(config, prompt_cache_blocks=True), seller, history, 'private note')
                    parts = cached['input'][1]['content']
                    decoded = json.loads(''.join(p['text'] for p in parts))
                    self.assertEqual(decoded, json.loads(compact['input'][1]['content']))
                    self.assertEqual(len(decoded['complete_market_history']), count)
                    self.assertEqual(cached['prompt_cache_options'], {'mode': 'explicit', 'ttl': '30m'})

    def test_complete_block_prefix_stays_identical_across_decisions(self):
        config = Config(action_interface='price', prompt_cache_blocks=True, gamma=1)
        history = records(51, config)
        earlier = prompt_body(config, 3, history[:50], 'old plan')['input'][1]['content']
        later = prompt_body(config, 3, history, 'new plan')['input'][1]['content']
        self.assertEqual(earlier[:-1], later[:len(earlier)-1])
        self.assertTrue(all(p.get('prompt_cache_breakpoint') == {'mode': 'explicit'} for p in earlier[:-1]))
        self.assertNotIn('prompt_cache_breakpoint', later[-2])  # Partial 51st row is not written.
        self.assertNotEqual(earlier[-1]['text'], later[-1]['text'])  # Current objective/note are dynamic.


if __name__ == '__main__':
    unittest.main()
