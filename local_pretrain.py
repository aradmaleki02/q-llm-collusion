"""Profit-only local experience before expensive LLM interaction; no API imports."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from capacity_audit import successor, teacher
from collusion.config import Config
from collusion.learner import Learner, VARIANTS
from collusion.market import outcome

ROOT = Path(__file__).resolve().parent


def opponents(mode, anchor, state, history, rng):
    if mode == 'static':
        price = anchor
    elif mode == 'matching':
        price = max(1.28, min(2.24, history[-1]['prices'][3])) if history else anchor
    elif mode == 'tolerant':
        price = anchor if state < 2 else 1.32
    else:
        price = rng.choice((1.28, 1.32, 1.36))
    return [price]*3


def evaluate(learner, cfg):
    results = []
    for mode in ('static', 'tolerant'):
        for anchor in (1.32, 1.80, 2.00, 2.20):
            rng, state, history = random.Random(900), 0, []
            for t in range(200):
                action = learner.act(history, False)
                price = cfg.prices[action]
                prices = opponents(mode, anchor, state, history, rng)+[price]
                demand, profit = outcome(prices, cfg)
                history.append(dict(prices=prices, demand=demand, profit=profit))
                state = successor(state, price, anchor)
            total = sum(r['profit'][3]*cfg.gamma**t for t, r in enumerate(history))
            if mode == 'static':
                best = max(outcome([anchor]*3+[p], cfg)[1][3] for p in cfg.prices)
                reference = best*(1-cfg.gamma**200)/(1-cfg.gamma)
            else:
                reference = float(teacher(cfg, anchor)[0].max())
            results.append(dict(mode=mode, anchor=anchor, discounted_profit=total,
                                oracle_value=reference, oracle_fraction=total/reference,
                                last50_mean_price=sum(r['prices'][3] for r in history[-50:])/50))
    return results


def run(args):
    torch.set_num_threads(1)
    cfg = replace(Config.load(ROOT/'config.json'), hidden_size=args.hidden, head_hidden_size=args.head,
                  device=args.device, experimental_seller=3, epsilon_start=1.)
    output = ROOT/'reports'/'local_pretrain'
    output.mkdir(parents=True, exist_ok=True)
    label = f'{args.tag}_' if args.tag else ''
    checkpoint = output/f'{label}{args.variant}_{args.hidden}_{args.head}.pt'
    learner = Learner(cfg, VARIANTS[args.variant], 1)
    rng = random.Random(401)
    if args.resume:
        previous = torch.load(checkpoint, map_location='cpu', weights_only=False)
        learner = previous['learner'].to(args.device)
        assert replace(learner.config, device=args.device) == cfg
        learner.config = cfg
        # Each extension starts a new, reproducibly seeded block of markets.
        # Existing replay episodes remain untouched under their original keys.
        rng.seed(401+learner.steps)
    start_step = learner.steps
    started = time.perf_counter()
    for step in range(learner.steps, learner.steps+args.steps):
        if step == start_step or step % 64 == 0:
            mode = rng.choice(args.modes)
            anchor = rng.choice((1.32, 1.52, 1.76, 2.04, 2.20))
            history, state = [], 0
        action = rng.randrange(cfg.price_count) if rng.random() < args.exploration else learner.act(history, True)
        price = cfg.prices[action]
        prices = opponents(mode, anchor, state, history, rng)+[price]
        demand, profit = outcome(prices, cfg)
        history.append(dict(prices=prices, demand=demand, profit=profit))
        learner.learn(f'local:{start_step}:{step//64}', history, action)
        state = successor(state, price, anchor)
        if (step+1) % 1000 == 0:
            print(json.dumps(dict(variant=args.variant, local_steps=step+1,
                gradient_updates=learner.updates, elapsed_seconds=time.perf_counter()-started)), flush=True)
    results = evaluate(learner, cfg)
    static = [r['oracle_fraction'] for r in results if r['mode']=='static']
    tolerant = [r['oracle_fraction'] for r in results if r['mode']=='tolerant']
    report = dict(variant=args.variant, config=cfg.dictionary(), steps=learner.steps,
                  gradient_updates=learner.updates, elapsed_seconds=time.perf_counter()-started,
                  api_calls=0, reward='Unmodified own profit',
                  training_modes=args.modes,
                  minimum_uniform_exploration=args.exploration,
                  provenance='Locally simulated static, matching, tolerant and competitive rivals; not an LLM surrogate fidelity claim or Keppo replication.',
                  readiness_rule='Mean static-oracle fraction >=.95, every static case >=.90; mean dynamic-oracle fraction >=.75.',
                  ready=sum(static)/len(static)>=.95 and min(static)>=.90 and sum(tolerant)/len(tolerant)>=.75,
                  results=results)
    # Preserve a resumable local policy; no supervised diagnostic weights enter this file.
    payload = dict(learner=learner, report=report, rng_state=rng.getstate(),
                   torch_rng=torch.get_rng_state(), market=(mode, anchor, history, state))
    torch.save(payload, checkpoint)
    torch.save(payload, checkpoint.with_stem(checkpoint.stem+f'_{learner.steps}'))
    encoded = json.dumps(report, indent=2)
    checkpoint.with_suffix('.json').write_text(encoded)
    checkpoint.with_stem(checkpoint.stem+f'_{learner.steps}').with_suffix('.json').write_text(encoded)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['ordinary', 'optimistic'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--head', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--exploration', type=float, default=0.)
    parser.add_argument('--modes', nargs='+', choices=['static', 'matching', 'tolerant', 'war'],
                        default=['static', 'matching', 'tolerant', 'war'])
    run(parser.parse_args())
