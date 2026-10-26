"""Local-only timing and representation checks. Never calls an LLM API."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

from collusion.config import Config
from collusion.learner import Learner, Network, observations
from collusion.market import outcome

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'reports' / 'capacity_audit'


def save(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def timing(head=0, widths=(32, 128, 256), allowed_devices=None):
    devices = [('cpu', 1), ('cpu', 4)] + [
        (f'cuda:{i}', 1) for i in range(torch.cuda.device_count())]
    if allowed_devices:
        devices = [(d, n) for d, n in devices if d in allowed_devices]
    result = dict(torch=torch.__version__, gpus=[torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())], rows=[], api_calls=0)
    base = replace(Config.load(ROOT / 'config.json'), experimental_seller=3, head_hidden_size=head)
    for hidden in widths:
        for length in (100, 500, 1000):
            for device, threads in devices:
                torch.set_num_threads(threads)
                cfg = replace(base, hidden_size=hidden, device=device)
                learner = Learner(cfg, 1., 123)
                prices = [1.8]*4
                demand, profit = outcome(prices, cfg)
                history = [dict(prices=prices, demand=demand, profit=profit)]*length
                learner.episodes = {'timing': history}
                learner.replay = [('timing', t, 25) for t in range(length//2, length)]
                sync = lambda: torch.cuda.synchronize(device) if device != 'cpu' else None
                learner._update()  # cuDNN/optimizer warm-up, excluded
                sync()
                elapsed = []
                for _ in range(5):
                    sync()
                    start = time.perf_counter()
                    learner._update()  # Includes encoding, transfers, 3 forwards, backward, Adam.
                    sync()
                    elapsed.append((time.perf_counter()-start)*1000)
                row = dict(hidden_size=hidden, parameters=sum(p.numel() for p in learner.online.parameters()),
                           head_hidden_size=head,
                           history_length=length, batch_size=cfg.batch_size, device=device,
                           cpu_threads=threads, median_update_ms=statistics.median(elapsed),
                           samples_ms=elapsed)
                result['rows'].append(row)
                save('timing.json' if not head else f'timing_head{head}.json', result)
                print(json.dumps(row), flush=True)
                del learner
    return result


# A finite-state, tolerant pricing opponent gives EXACT dynamic-programming Q*.
# States 0/1 are cooperation with zero/one strike; 2/3/4 are 1/2/3 punishment rounds.
# Histories reveal prices and own demand/profit only, as in the paid experiment.
def successor(state, price, anchor):
    if state >= 2:
        return state-1 if state > 2 else 0
    return (1 if state == 0 else 4) if price < anchor-.04 else 0


def teacher(cfg, anchor):
    reward = torch.empty(5, cfg.price_count)
    next_states = torch.empty(5, cfg.price_count, dtype=torch.long)
    for state in range(5):
        rival = anchor if state < 2 else 1.32
        for i, price in enumerate(cfg.prices):
            reward[state, i] = outcome([rival]*3+[price], cfg)[1][3]
            next_states[state, i] = successor(state, price, anchor)
    value = torch.zeros(5)
    for _ in range(1000):
        q = reward+cfg.gamma*value[next_states]
        new = q.max(1).values
        if (new-value).abs().max() < 1e-6:
            break
        value = new
    return reward+cfg.gamma*value[next_states]


def dataset(cfg, seed, count, lengths):
    rng = random.Random(seed)
    anchors = (1.60, 1.76, 1.92, 2.08, 2.24)
    tables = {a: teacher(cfg, a) for a in anchors}
    sequences, targets = [], []
    for _ in range(count):
        anchor, length = rng.choice(anchors), rng.choice(lengths)
        state, history = 0, []
        for t in range(length):
            rival = anchor if state < 2 else 1.32
            price = rng.choice(cfg.prices) if rng.random() < .35 else anchor
            prices = [rival]*3+[price]
            demand, profit = outcome(prices, cfg)
            history.append(dict(prices=prices, demand=demand, profit=profit))
            state = successor(state, price, anchor)
        sequences.append(observations(history, cfg))
        targets.append(tables[anchor][state])
    return sequences, torch.stack(targets)


def score(network, data, center, scale):
    sequences, truth = data
    predicted = []
    with torch.no_grad():
        for start in range(0, len(sequences), 16):
            predicted.append(network(sequences[start:start+16]).cpu()*scale+center)
    predicted = torch.cat(predicted)
    chosen = predicted.argmax(1)
    regret = truth.max(1).values-truth.gather(1, chosen[:, None]).squeeze(1)
    span = truth.max(1).values-truth.min(1).values
    return dict(q_rmse=float(F.mse_loss(predicted, truth).sqrt()),
                greedy_value_regret=float(regret.mean()),
                normalized_regret_pct=float((regret/span.clamp_min(1e-6)).mean()*100),
                exact_best_action_pct=float((chosen == truth.argmax(1)).float().mean()*100))


def capacity(device, steps, head=0, widths=(32, 128, 256)):
    torch.set_num_threads(1)
    cfg = replace(Config.load(ROOT / 'config.json'), experimental_seller=3, head_hidden_size=head)
    filename = f'capacity_head{head}_{steps}.json'
    train = dataset(cfg, 101, 1024, (8, 32, 128, 256))
    validation = dataset(cfg, 202, 256, (8, 32, 128, 256, 1000))
    test = dataset(cfg, 303, 256, (8, 32, 128, 256, 1000))
    center, scale = train[1].mean(), train[1].std()
    result = dict(api_calls=0, device=device, training_updates=steps,
        task='Fit exact discounted Q* against a scripted tolerant finite-state opponent; diagnostic only, no live-policy weight reuse.',
        head_hidden_size=head,
        limitations='Tests representation and optimization on this surrogate, not real LLM responses or convergence from 1000 paid transitions.',
        selection_rule='Smallest width with mean validation normalized greedy regret <=2%; use test only after selection.', rows=[])
    for hidden in widths:
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            rng = random.Random(seed)
            network = Network(replace(cfg, hidden_size=hidden)).to(device)
            optimizer = torch.optim.Adam(network.parameters(), lr=.001)
            losses = []
            started = time.perf_counter()
            for step in range(steps):
                ids = rng.sample(range(len(train[0])), 16)
                predicted = network([train[0][i] for i in ids])
                target = ((train[1][ids]-center)/scale).to(device)
                loss = F.mse_loss(predicted, target)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            row = dict(hidden_size=hidden, seed=seed, parameters=sum(p.numel() for p in network.parameters()),
                       elapsed_seconds=time.perf_counter()-started,
                       initial_loss=statistics.mean(losses[:25]), final_loss=statistics.mean(losses[-25:]),
                       training=score(network, train, center, scale),
                       validation=score(network, validation, center, scale))
            # Save only diagnostic weights. They are never loaded into the live Q study.
            OUTPUT.mkdir(parents=True, exist_ok=True)
            torch.save(network.state_dict(), OUTPUT / f'diagnostic_{hidden}_{head}_{seed}.pt')
            result['rows'].append(row)
            save(filename, result)
            print(json.dumps(row), flush=True)
    passing = [h for h in widths if statistics.mean(
        r['validation']['normalized_regret_pct'] for r in result['rows'] if r['hidden_size']==h) <= 2]
    selected = min(passing) if passing else None
    result['selected_hidden_size'] = selected
    if selected:
        result['heldout_test'] = []
        for seed in (1, 2, 3):
            network = Network(replace(cfg, hidden_size=selected)).to(device)
            network.load_state_dict(torch.load(OUTPUT / f'diagnostic_{selected}_{head}_{seed}.pt', map_location=device, weights_only=True))
            result['heldout_test'].append(dict(seed=seed, **score(network, test, center, scale)))
    save(filename, result)
    print(json.dumps({'selected_hidden_size': selected, 'heldout_test': result.get('heldout_test')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['timing', 'capacity'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--widths', type=int, nargs='+', default=[32, 128, 256])
    parser.add_argument('--benchmark-devices', nargs='+')
    args = parser.parse_args()
    timing(args.head, args.widths, args.benchmark_devices) if args.mode == 'timing' else capacity(args.device, args.steps, args.head, args.widths)
