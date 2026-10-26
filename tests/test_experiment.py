import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import torch

from collusion.api import Gateway, parse_decision, prompt_body, reservation, response_cost
from collusion.config import Config
from collusion.engine import make_experiment, make_probe, run_experiment
from collusion.learner import hysteretic_loss, observations, td_targets
from collusion.market import benchmarks, outcome
from collusion.report import write_report
from collusion.storage import Paused, Store


def response(index=20, **overrides):
    value = {"id": "resp_test", "status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"price_index": index, "strategy": "test"})}]}],
        "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 100}}
    value.update(overrides)
    return value


def history_row(t, prices=None):
    prices = prices or [1.4, 1.6, 1.8, 2.0]
    demand, profit = outcome(prices, Config())
    return {"round": t, "prices": prices, "demand": demand, "profit": profit,
            "actions": [10, 15, 20, 25], "strategies": [""] * 4, "loss": None}


class MarketTests(unittest.TestCase):
    def test_conservation_and_profit(self):
        c = Config()
        d, p = outcome([1.7] * 4, c)
        self.assertLess(sum(d), 1)
        self.assertAlmostEqual(p[0], 0.7 * d[0])
        self.assertEqual(d, [d[0]] * 4)
        self.assertEqual(outcome([1.0] * 4, c)[1], [0.0] * 4)
        self.assertGreater(outcome([1.5, 1.7, 1.7, 1.7], c)[0][0], d[0])

    def test_four_seller_benchmarks_and_best_responses(self):
        c = Config()
        b = benchmarks(c)
        self.assertAlmostEqual(b["continuous_nash_price"], 1.33146069, places=7)
        self.assertAlmostEqual(b["continuous_joint_price"], 2.05441118, places=7)
        for equilibrium in b["symmetric_grid_nash"]:
            p = equilibrium["price"]
            self.assertTrue(all(outcome([a, p, p, p], c)[1][0] <= equilibrium["profit"] + 1e-12
                                for a in c.prices))

    def test_finite_vs_truncation_bootstrap(self):
        rewards, next_q = torch.tensor([2., 2.]), torch.tensor([10., 10.])
        self.assertTrue(torch.equal(td_targets(rewards, next_q, torch.tensor([False, True]), 1),
                                    torch.tensor([12., 2.])))
        self.assertAlmostEqual(float(td_targets(rewards[:1], next_q[:1], torch.tensor([False]), .95)[0]), 11.5)

    def test_optimism_changes_only_negative_td_weight(self):
        predicted = torch.tensor([0., 0.], requires_grad=True)
        targets = torch.tensor([2., -2.])
        hysteretic_loss(predicted, targets, .25).backward()
        self.assertAlmostEqual(abs(float(predicted.grad[1] / predicted.grad[0])), .25)

    def test_full_history_no_other_profit_leakage(self):
        c = Config(rounds=600)
        history = [history_row(t) for t in range(1, 501)]
        tensor = observations(history, c)
        self.assertEqual(tuple(tensor.shape), (501, 7))
        original = tensor.clone()
        history[0]["profit"][1] = 9999
        self.assertTrue(torch.equal(observations(history, c), original))
        body = prompt_body(c, 0, history, "private")
        payload = json.loads(body["input"][1]["content"])
        self.assertEqual(len(payload["complete_market_history"]), 500)
        self.assertEqual(len(payload["complete_market_history"][0]), 7)
        self.assertNotIn(9999, payload["complete_market_history"][0])
        self.assertNotIn("remain", payload["objective"])
        finite = json.loads(prompt_body(Config(rounds=600, gamma=1), 0, history, "")["input"][1]["content"])
        self.assertIn("100 rounds remain", finite["objective"])


class DurableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "runs")
        self.config = Config(rounds=4, warmup_transitions=1, batch_size=2, hidden_size=8)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def create(self, backend="mock", variants=None):
        return make_experiment(self.store, "test", self.config, backend,
                               variants or ["ordinary"], [1], 1, 1)

    def test_budget_is_shared_and_reservations_survive(self):
        self.store.budget(.1)
        first = self.store.reserve("a", "x", {}, .06)
        with self.assertRaises(Paused):
            self.store.reserve("b", "y", {}, .05)
        self.store.mark(first, "uncertain")
        self.assertAlmostEqual(self.store.spending()["reserved"], .06)
        with self.assertRaises(ValueError):
            self.store.retry_request(first, False)
        self.store.retry_request(first, True)
        self.assertAlmostEqual(self.store.spending()["charged"], .06)
        self.assertEqual(self.store.spending()["reserved"], 0)

    def test_partial_round_reuses_paid_response_and_keeps_simultaneity(self):
        self.create("openai")
        quota = HTTPError("url", 429, "quota", {}, io.BytesIO(json.dumps({
            "error": {"code": "insufficient_quota"}}).encode()))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(Gateway, "_send", side_effect=[response(), quota]) as send:
            self.assertTrue(run_experiment(self.store, "test", progress=lambda _: None).startswith("paused"))
            self.assertEqual(send.call_count, 2)
        _, state, _ = self.store.load("test")
        pending_action = state["market"]["pending"]["q_action"]
        self.assertEqual(state["committed_rounds"], 0)
        bodies = []
        def send_again(body):
            bodies.append(body)
            return response()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(Gateway, "_send", side_effect=send_again) as send:
            run_experiment(self.store, "test", progress=lambda _: None, step_limit=1)
            self.assertEqual(send.call_count, 2)  # Only sellers two and three.
        row = json.loads(self.store.records("test")[0]["record"])
        self.assertEqual(row["actions"][0], pending_action)
        self.assertTrue(all(not json.loads(b["input"][1]["content"])["complete_market_history"] for b in bodies))
        self.assertAlmostEqual(self.store.spending()["charged"], 3 * .00015)

    def test_invalid_response_is_charged_and_requires_explicit_retry(self):
        self.create("openai")
        gateway = Gateway(self.store, self.config, "test")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(gateway, "_send", return_value=response(status="incomplete")) as send:
            with self.assertRaises(Paused):
                gateway.decide("0/1/1", 1, [], "")
            with self.assertRaises(Paused):
                gateway.decide("0/1/1", 1, [], "")
            self.assertEqual(send.call_count, 1)
        row = self.store.request("test", "0/1/1")
        self.assertGreater(row["charged"], 0)
        self.store.retry_request(row["id"], False, 8192)
        with patch.object(gateway, "_send", return_value=response()) as send:
            gateway.decide("0/1/1", 1, [], "")
            self.assertEqual(send.call_args.args[0]["max_output_tokens"], 8192)
        with patch.object(gateway, "_send") as send:
            gateway.decide("0/1/1", 1, [], "")
            send.assert_not_called()

    def test_timeout_is_not_silently_retried(self):
        self.create("openai")
        gateway = Gateway(self.store, self.config, "test")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(gateway, "_send", side_effect=TimeoutError) as send:
            with self.assertRaises(Paused):
                gateway.decide("x", 1, [], "")
            with self.assertRaises(Paused):
                gateway.decide("x", 1, [], "")
            self.assertEqual(send.call_count, 1)
        self.assertEqual(self.store.request("test", "x")["status"], "uncertain")

    def test_budget_pause_makes_no_network_call_and_resumes(self):
        self.create("openai")
        self.store.budget(.00000001)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-a-real-key"}), \
             patch.object(Gateway, "_send", return_value=response()) as send:
            run_experiment(self.store, "test", progress=lambda _: None)
            send.assert_not_called()
            self.store.budget(50)
            run_experiment(self.store, "test", progress=lambda _: None, step_limit=1)
            self.assertEqual(send.call_count, 3)
        self.assertEqual(len(self.store.records("test")), 1)

    def test_checkpoint_resume_matches_uninterrupted_and_reports(self):
        self.create(variants=["ordinary", "optimistic"])
        run_experiment(self.store, "test", progress=lambda _: None, step_limit=3)
        run_experiment(self.store, "test", progress=lambda _: None)
        reference = Store(Path(self.temp.name) / "reference")
        try:
            make_experiment(reference, "test", self.config, "mock", ["ordinary", "optimistic"], [1], 1, 1)
            run_experiment(reference, "test", progress=lambda _: None)
            self.assertEqual([r["record"] for r in self.store.records("test")],
                             [r["record"] for r in reference.records("test")])
            _, resumed, status = self.store.load("test")
            _, straight, _ = reference.load("test")
            self.assertEqual(status, "complete")
            for key in resumed["learners"]:
                for parameter, value in resumed["learners"][key].online.state_dict().items():
                    self.assertTrue(torch.equal(value, straight["learners"][key].online.state_dict()[parameter]))
            report = write_report(self.store, "test", Path(self.temp.name) / "report")
            self.assertIn("OFFLINE MOCK DATA", (report / "report.html").read_text())
            self.assertTrue((report / "metrics.csv").is_file())
            make_probe(self.store, "test", "probe", "optimistic", 1, 1, 2, 2)
            run_experiment(self.store, "probe", progress=lambda _: None)
            self.assertEqual(len(self.store.records("probe")), 6)
            _, probe, _ = self.store.load("probe")
            self.assertEqual(probe["learners"]["optimistic:1"].steps, resumed["learners"]["optimistic:1"].steps)
        finally:
            reference.close()

    def test_evaluation_does_not_update_weights_or_replay(self):
        self.create()
        run_experiment(self.store, "test", progress=lambda _: None, step_limit=4)
        _, state, _ = self.store.load("test")
        weights = {k: v.clone() for k, v in state["learners"]["ordinary:1"].online.state_dict().items()}
        steps = state["learners"]["ordinary:1"].steps
        run_experiment(self.store, "test", progress=lambda _: None)
        _, state, _ = self.store.load("test")
        learner = state["learners"]["ordinary:1"]
        self.assertEqual(learner.steps, steps)
        self.assertTrue(all(torch.equal(value, learner.online.state_dict()[key]) for key, value in weights.items()))

    def test_cost_counts_reasoning_cache_and_long_context(self):
        c = self.config
        result = response(usage={"input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
            "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 490}})
        self.assertAlmostEqual(response_cost(result, c, 1), (700*.1 + 200*.01 + 100*.125 + 500*.5)/1e6)
        short = prompt_body(c, 1, [], "")
        long = dict(short, input=[{"role": "user", "content": "a" * 300000}])
        self.assertGreater(reservation(long, c), reservation(short, c))


if __name__ == "__main__":
    unittest.main()
