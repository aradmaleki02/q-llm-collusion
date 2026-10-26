import copy
import json
from concurrent.futures import ThreadPoolExecutor

import torch

from .api import Gateway
from .config import Config
from .learner import Learner, VARIANTS
from .market import outcome
from .storage import Paused, Store


def seed_number(seed, variant):
    # Same network initialization and exploration stream across matched Q variants.
    # API sampling is not seed-controlled and is not claimed to be reproducible.
    return int(seed)


def make_experiment(store, name, config, backend, variants, seeds, train_episodes, eval_episodes,
                    include_llm_reference=True):
    if store.exists(name):
        raise ValueError(f"{name!r} already exists. Use resume, or choose a different --name.")
    if not variants or any(v not in VARIANTS for v in variants):
        raise ValueError("Choose ordinary, optimistic and/or strong")
    if len(set(variants)) != len(variants) or len(set(seeds)) != len(seeds):
        raise ValueError("Variants and seeds must not contain duplicates")
    if train_episodes < 1 or eval_episodes < 1 or not seeds:
        raise ValueError("Require positive training/evaluation counts and at least one seed")
    tasks = []
    for episode in range(1, train_episodes + 1):
        for seed in seeds:
            for variant in variants:
                tasks.append(dict(variant=variant, seed=seed, phase="train", episode=episode))
    for episode in range(1, eval_episodes + 1):
        for seed in seeds:
            for variant in [*variants, *(["all_llm"] if include_llm_reference else [])]:
                tasks.append(dict(variant=variant, seed=seed, phase="eval", episode=episode))
    learners = {f"{variant}:{seed}": Learner(config, VARIANTS[variant], seed_number(seed, variant))
                for seed in seeds for variant in variants}
    state = {"backend": backend, "tasks": tasks, "task_index": 0, "market": None,
             "learners": learners, "committed_rounds": 0,
             "torch_rng": torch.get_rng_state(), "version": 1}
    store.save(name, config, state, "ready")
    return state


def make_baseline(store, name, config, backend="openai", episodes=1):
    if store.exists(name) or episodes < 1:
        raise ValueError("Choose a new name and a positive baseline episode count")
    state = {"backend": backend, "tasks": [dict(variant="all_llm", seed=i, phase="eval", episode=1)
             for i in range(1, episodes + 1)], "task_index": 0, "market": None,
             "learners": {}, "committed_rounds": 0, "torch_rng": torch.get_rng_state(), "version": 1}
    store.save(name, config, state, "ready")


def run_experiment(store, name, progress=print, step_limit=None, parallel_calls=None, stop_after_task=False):
    values, state, status = store.load(name)
    if state.get('parallel_execution'):
        raise Paused('This checkpoint has parallel markets; resume with study_controller.py.')
    if status.startswith('archived') or state.get('excluded_from_primary_analysis'):
        raise ValueError('This archived interface-audit run is preserved for review and cannot be resumed.')
    config = Config(**values).validate()
    for learner in state["learners"].values():
        learner.to(config.device)
    torch.set_num_threads(1)
    torch.set_rng_state(state["torch_rng"])
    store.recover_inflight(name)
    gateway = Gateway(store, config, name, state["backend"], spend_ceiling=state.get('spend_ceiling'))
    workers = config.parallel_llm_calls if parallel_calls is None else parallel_calls
    if not 1 <= workers <= 4:
        raise ValueError("parallel_calls must be between one and four")
    committed = 0

    def save(status="running", row=None):
        state["torch_rng"] = torch.get_rng_state()
        store.save(name, config, state, status, row)

    try:
        while state["task_index"] < len(state["tasks"]):
            if (store.root / 'pause_requested').exists():
                raise Paused('Pause requested between rounds; remove runs/pause_requested before resuming.')
            ti = state["task_index"]
            task = state["tasks"][ti]
            if state["market"] is None:
                prefix = task.get("prefix", [])
                state["market"] = {"history": copy.deepcopy(prefix),
                    "strategies": task.get("initial_strategies", [""] * 4).copy(), "pending": None}
                save()
            market = state["market"]
            history = market["history"]
            round_number = len(history) + 1
            end = task.get("end_round", config.rounds)
            if round_number > end:
                state["task_index"] += 1
                state["market"] = None
                save()
                if stop_after_task:
                    return 'task_complete'
                continue
            training = task["phase"] == "train"
            learner = state["learners"].get(f"{task['variant']}:{task['seed']}")
            if market["pending"] is None:
                q_action = learner.act(history, training) if learner else None
                if "forced_index" in task and round_number == len(task["prefix"]) + 1:
                    q_action = task["forced_index"]
                # Persist action and RNG after selection before the first paid call.
                market["pending"] = {"q_action": q_action}
                save()
            actions = [None] * 4
            strategies = market["strategies"].copy()
            if learner:
                actions[config.experimental_seller] = market["pending"]["q_action"]
            sellers = [i for i in range(4) if not learner or i != config.experimental_seller]

            def decide_seller(seller):
                if workers == 1:
                    return gateway.decide(f"{ti}/{round_number}/{seller}", seller, history, strategies[seller])
                # Each HTTP worker owns its SQLite connection. Reservation uses
                # BEGIN IMMEDIATE so simultaneous requests cannot overspend.
                worker_store = Store(store.root)
                try:
                    worker_gateway = Gateway(worker_store, config, name, state["backend"],
                                             spend_ceiling=state.get('spend_ceiling'))
                    return worker_gateway.decide(f"{ti}/{round_number}/{seller}", seller, history, strategies[seller])
                finally:
                    worker_store.close()

            if workers > 1:
                with ThreadPoolExecutor(max_workers=min(workers, len(sellers))) as pool:
                    decisions = list(pool.map(decide_seller, sellers))
            else:
                decisions = [decide_seller(seller) for seller in sellers]
            for seller, decision in zip(sellers, decisions):
                actions[seller] = decision["index"]
                strategies[seller] = decision["strategy"]
            prices = [config.prices[index] for index in actions]
            demand, profit = outcome(prices, config)
            record = {"round": round_number, "prices": prices, "actions": actions,
                      "demand": demand, "profit": profit, "strategies": strategies, "loss": None}
            history.append(record)
            if training and learner:
                record["loss"] = learner.learn(f"task:{ti}", history, actions[config.experimental_seller])
            market["strategies"] = strategies
            market["pending"] = None
            state["committed_rounds"] += 1
            # Round output and the post-update network enter the SAME transaction.
            save(row=(ti, task, record))
            committed += 1
            if round_number == 1 or round_number % 25 == 0 or round_number == end:
                spent = store.spending()
                prefix = "MOCK " if state["backend"] == "mock" else ""
                progress(f"{prefix}{name} | {task['variant']} seed={task['seed']} {task['phase']} "
                         f"game={task['episode']} round={round_number}/{end} "
                         f"gradient_updates={learner.updates if learner else 0} | "
                         f"prices={','.join(f'{p:.2f}' for p in prices)} | "
                         f"spent=${spent['charged']:.3f}/${spent['budget']:.2f}")
            if step_limit and committed >= step_limit:
                save("paused: requested step limit")
                return "paused: requested step limit"
        save("complete")
        return "complete"
    except (Paused, KeyboardInterrupt) as exc:
        # Reload the last durable state: interruption during a gradient update must
        # not commit an appended history with an incomplete optimizer update.
        _, durable, _ = store.load(name)
        reason = str(exc) if isinstance(exc, Paused) else "Interrupted by Ctrl+C; resume to continue."
        store.save(name, config, durable, "paused: " + reason)
        progress("PAUSED: " + reason)
        return "paused: " + reason


def make_probe(store, source_name, name, variant, seed, episode, at_round, steps):
    values, source, _ = store.load(source_name)
    config = Config(**values)
    candidates = [(i, t) for i, t in enumerate(source["tasks"])
                  if t["variant"] == variant and t["seed"] == seed and t["episode"] == episode
                  and t["phase"] == "eval"]
    if not candidates:
        raise ValueError("No matching held-out evaluation market")
    ti, task = candidates[0]
    records = [json.loads(row["record"]) for row in store.records(source_name, ti)]
    if len(records) != config.rounds:
        raise ValueError("Complete the source evaluation episode before branching it")
    if variant == "all_llm" or not 1 <= at_round < config.rounds or steps < 1:
        raise ValueError("Choose a Q variant, a valid branch round, and positive steps")
    if store.exists(name):
        raise ValueError("Probe name already exists; use resume")
    prefix = records[:at_round]
    tasks = []
    reference_price = prefix[-1]["prices"][config.experimental_seller]
    for branch, scale in (("control", 1.0), ("undercut", 0.9), ("increase", 1.1)):
        new = dict(variant=variant, seed=seed, phase=f"probe_{branch}", episode=episode,
                   prefix=prefix, initial_strategies=prefix[-1]["strategies"],
                   end_round=min(config.rounds, at_round + steps))
        if branch != "control":
            new["forced_index"] = min(range(config.price_count),
                key=lambda i: abs(config.prices[i] - reference_price * scale))
        tasks.append(new)
    key = f"{variant}:{seed}"
    learner = copy.deepcopy(source["learners"][key])
    learner.replay, learner.episodes = [], {}
    state = {"backend": source["backend"], "tasks": tasks, "task_index": 0, "market": None,
             "learners": {key: learner}, "committed_rounds": 0,
             "torch_rng": source["torch_rng"], "version": 1, "probe_source": source_name}
    store.save(name, config, state, "ready")
