import argparse
import json
import re
import sys

import torch

from .config import Config
from .engine import make_baseline, make_experiment, make_probe, run_experiment
from .market import benchmarks
from .report import write_report
from .storage import Paused, Store, single_writer


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise argparse.ArgumentTypeError("Use a short name with letters, digits, underscores and hyphens")
    return value


def parser():
    root = argparse.ArgumentParser(description="Four-seller pricing research experiment")
    root.add_argument("--runs-dir", default="runs", help="Shared checkpoint and spending-ledger directory")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start a NEW training/evaluation suite")
    run.add_argument("--name", type=safe_name, default="pilot")
    run.add_argument("--config", default="config.json")
    run.add_argument("--backend", choices=["openai", "mock"], default="openai")
    run.add_argument("--variants", nargs="+", choices=["ordinary", "optimistic", "strong"],
                     default=["ordinary", "optimistic"])
    run.add_argument("--seeds", nargs="+", type=int, default=[1])
    run.add_argument("--train-episodes", type=int, default=1)
    run.add_argument("--eval-episodes", type=int, default=1)
    run.add_argument("--rounds", type=int)
    run.add_argument("--gamma", type=float, choices=[0.95, 0.99, 1.0])
    run.add_argument("--budget", type=float, help="Set total SHARED lifetime allowance, not additional dollars")
    run.add_argument("--steps", type=int, help="Pause after this many newly committed rounds")
    run.add_argument("--parallel-calls", type=int, choices=[1, 2, 3, 4])
    baseline = commands.add_parser("baseline", help="Run only the four-LLM reference before training")
    baseline.add_argument("--name", type=safe_name, default="llm-baseline")
    baseline.add_argument("--config", default="config.json")
    baseline.add_argument("--backend", choices=["openai", "mock"], default="openai")
    baseline.add_argument("--episodes", type=int, default=1)
    baseline.add_argument("--rounds", type=int)
    baseline.add_argument("--gamma", type=float, choices=[0.95, 0.99, 1.0])
    baseline.add_argument("--budget", type=float)
    baseline.add_argument("--steps", type=int)
    baseline.add_argument("--parallel-calls", type=int, choices=[1, 2, 3, 4])
    resume = commands.add_parser("resume", help="Continue saved state without changing experimental settings")
    resume.add_argument("--name", type=safe_name, default="pilot")
    resume.add_argument("--budget", type=float)
    resume.add_argument("--steps", type=int)
    resume.add_argument("--parallel-calls", type=int, choices=[1, 2, 3, 4])
    status = commands.add_parser("status", help="Show checkpoints, spending, and unresolved requests")
    status.add_argument("--name", type=safe_name)
    report = commands.add_parser("report", help="Export CSV metrics, raw rounds, and an HTML chart report")
    report.add_argument("--name", type=safe_name, default="pilot")
    report.add_argument("--output")
    commands.add_parser("benchmarks", help="Print four-seller continuous and grid benchmarks").add_argument(
        "--config", default="config.json")
    inspect = commands.add_parser("inspect-request", help="Review an API call and its saved response locally")
    inspect.add_argument("request_id", type=int)
    retry = commands.add_parser("retry-request", help="Release an unusable/uncertain response for a new attempt")
    retry.add_argument("request_id", type=int)
    retry.add_argument("--accept-possible-duplicate", action="store_true")
    retry.add_argument("--max-output-tokens", type=int)
    probe = commands.add_parser("probe", help="Branch a completed evaluation into control/cut/increase markets")
    probe.add_argument("--source", default="pilot", type=safe_name)
    probe.add_argument("--name", default="pilot-probe", type=safe_name)
    probe.add_argument("--variant", choices=["ordinary", "optimistic", "strong"], default="optimistic")
    probe.add_argument("--seed", type=int, default=1)
    probe.add_argument("--episode", type=int, default=1)
    probe.add_argument("--at-round", type=int, default=250)
    probe.add_argument("--probe-rounds", type=int, default=30)
    probe.add_argument("--budget", type=float)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(1)
    if getattr(args, "steps", None) is not None and args.steps < 1:
        parser().error("--steps must be positive")
    if args.command == "benchmarks":
        print(json.dumps(benchmarks(Config.load(args.config)), indent=2))
        return 0
    store = None
    try:
        with single_writer(args.runs_dir):
            store = Store(args.runs_dir)
            if getattr(args, "budget", None) is not None:
                store.budget(args.budget)
            if args.command in ("run", "baseline"):
                config = Config.load(args.config, rounds=args.rounds, gamma=args.gamma)
                if config.model != "gpt-6-luna":
                    raise ValueError("This version's verified prices/reservations support gpt-6-luna only. "
                                     "Do not substitute a model without updating and verifying billing rules.")
                if args.command == "baseline":
                    make_baseline(store, args.name, config, args.backend, args.episodes)
                else:
                    make_experiment(store, args.name, config, args.backend, args.variants, args.seeds,
                                    args.train_episodes, args.eval_episodes)
                print(f"Created {args.name}. Shared budget ${store.budget():.2f}. "
                      "Training is resumable; no live calls occur in mock mode.")
            elif args.command == "probe":
                make_probe(store, args.source, args.name, args.variant, args.seed,
                           args.episode, args.at_round, args.probe_rounds)
            if args.command in ("run", "baseline", "resume", "probe"):
                result = run_experiment(store, args.name, step_limit=getattr(args, "steps", None),
                                        parallel_calls=getattr(args, "parallel_calls", None))
                print(f"Status: {result}")
                print(f"Report: {write_report(store, args.name) / 'report.html'}")
            elif args.command == "report":
                print(write_report(store, args.name, args.output) / "report.html")
            elif args.command == "status":
                print(json.dumps(store.spending(), indent=2))
                for item in store.db.execute("SELECT name,status,updated FROM experiments ORDER BY updated DESC"):
                    if args.name is None or item["name"] == args.name:
                        _, state, _ = store.load(item["name"])
                        print(f"{item['name']}: {item['status']} | backend={state['backend']} | "
                              f"market {min(state['task_index']+1, len(state['tasks']))}/{len(state['tasks'])} | "
                              f"{state['committed_rounds']} committed rounds")
                for item in store.db.execute("SELECT id,experiment,status,error FROM requests "
                                             "WHERE status IN ('pending','uncertain','invalid')"):
                    print(dict(item))
            elif args.command == "inspect-request":
                row = store.db.execute("SELECT * FROM requests WHERE id=?", (args.request_id,)).fetchone()
                if not row:
                    raise ValueError("Unknown request id")
                print(json.dumps(dict(row), indent=2))  # Never contains Authorization or the key.
            elif args.command == "retry-request":
                store.retry_request(args.request_id, args.accept_possible_duplicate, args.max_output_tokens)
                print("Released for retry. Prior costs remain recorded. Run resume to continue.")
        return 0
    except Paused as exc:
        print(f"PAUSED: {exc}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    sys.exit(main())
