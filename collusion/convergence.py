"""Adaptive multi-game training. Convergence is operational, never an optimality proof."""
import copy
import json
import statistics
import time

import torch

from .config import Config
from .learner import observations
from .storage import Paused


def relative_change(a, b):
    return abs(a-b)/max(abs(a), abs(b), 1e-8)


def plateau(entries, policy):
    window = policy['return_window_games']
    if len(entries) < max(policy['minimum_games'], 2*window):
        return False
    earlier, latest = entries[-2*window:-window], entries[-window:]
    returns_stable = all(relative_change(statistics.mean(r[key] for r in earlier),
        statistics.mean(r[key] for r in latest)) <= policy['return_relative_tolerance']
        for key in ('total_profit', 'discounted_profit'))
    row = entries[-1]
    return returns_stable and row['policy_change_fraction'] is not None and (
        row['policy_change_fraction'] <= policy['policy_change_tolerance'] and
        row['q_relative_drift'] <= policy['q_drift_tolerance'])


def training_stop_reason(state, available, policy):
    monitors = state['training_convergence']['monitors']
    complete = len(monitors) == len(state['learners'])
    if complete and all(m['history'][-1]['converged'] for m in monitors.values()):
        return 'operational_convergence'
    costs = [r['api_cost_usd'] for m in monitors.values() for r in m['history']]
    next_pair_cost = statistics.mean(costs[-6:])*len(state['learners']) if costs else 0
    if available < next_pair_cost+policy['evaluation_reserve_usd']:
        return 'budget_limited_not_converged'
    return None


def record_game(learner, history, task, policy, monitor, api_cost):
    learner.to(learner.config.device)
    # More optimization of actual paid experience; these updates make no API calls.
    for _ in range(policy['extra_replay_updates_per_game'] if learner.replay else 0):
        learner._update()
    if len(monitor['history']) < policy['probe_source_games']:
        count = policy['probe_prefixes_per_game']
        ends = sorted({round(i*(len(history)-1)/(count-1)) for i in range(count)})
        monitor['probes'].extend(observations(history[:end], learner.config) for end in ends)
    with torch.no_grad():
        values = torch.cat([learner.online(monitor['probes'][i:i+8]).cpu()
                            for i in range(0, len(monitor['probes']), 8)])
    previous = monitor.get('values')
    same_bank = previous is not None and previous.shape == values.shape
    entry = dict(episode=task['episode'], rounds=len(history), api_cost_usd=api_cost,
        total_profit=sum(r['profit'][learner.config.experimental_seller] for r in history),
        discounted_profit=sum(learner.config.gamma**t*r['profit'][learner.config.experimental_seller]
                              for t, r in enumerate(history)),
        policy_change_fraction=float((values.argmax(1) != previous.argmax(1)).float().mean()) if same_bank else None,
        q_relative_drift=float((values-previous).abs().mean()/previous.abs().mean().clamp_min(1e-8)) if same_bank else None,
        environment_steps=learner.steps, gradient_updates=learner.updates)
    monitor['history'].append(entry)
    monitor['values'] = values
    monitor['stable_checks'] = monitor.get('stable_checks', 0)+1 if plateau(monitor['history'], policy) else 0
    entry['consecutive_stable_checks'] = monitor['stable_checks']
    entry['converged'] = monitor['stable_checks'] >= policy['required_stable_checks']


def advance_training(store, name, policy):
    """Record completed training games, then insert another matched pair before untouched evals."""
    values, state, status = store.load(name)
    cfg = Config(**values)
    if 'training_convergence' not in state:
        # Completed/started evaluation cannot be relabeled or moved into training.
        if any(t['phase'] != 'train' for t in state['tasks'][:state['task_index']]):
            raise Paused('Evaluation has already started; use a new development study for more training.')
        state['training_convergence'] = dict(policy=copy.deepcopy(policy), recorded_tasks=[], monitors={})
        state.setdefault('training_plan_amendments', []).append(dict(
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            applied_after_rounds=state['committed_rounds'],
            previous_tasks=copy.deepcopy(state['tasks']),
            reason=policy['authorization']))
    training = state['training_convergence']
    if training['policy'] != policy:
        raise Paused('Saved convergence criteria differ; review an explicit protocol amendment before changing them.')
    torch.set_rng_state(state['torch_rng'])
    for ti, task in enumerate(state['tasks'][:state['task_index']]):
        if task['phase'] != 'train' or ti in training['recorded_tasks']:
            continue
        records = [json.loads(r['record']) for r in store.records(name, ti)]
        if len(records) != cfg.rounds:
            raise Paused('Only completed market games can contribute to convergence.')
        key = f"{task['variant']}:{task['seed']}"
        monitor = training['monitors'].setdefault(key, dict(history=[], probes=[]))
        cost = store.db.execute('SELECT COALESCE(SUM(charged),0) FROM requests WHERE experiment=? '
                                'AND position LIKE ?', (name, f'{ti}/%')).fetchone()[0]
        record_game(state['learners'][key], records, task, policy, monitor, cost)
        training['recorded_tasks'].append(ti)
    ti = state['task_index']
    if ti < len(state['tasks']) and state['tasks'][ti]['phase'] == 'eval' and not state.get('training_stop_reason'):
        reason = training_stop_reason(state, store.spending()['available'], policy)
        if reason:
            state['training_stop_reason'] = reason
        else:
            # No request may exist for a future task whose index is about to move.
            for index in range(ti, len(state['tasks'])):
                count = store.db.execute('SELECT count(*) FROM requests WHERE experiment=? AND position LIKE ?',
                                         (name, f'{index}/%')).fetchone()[0]
                if count:
                    raise Paused('Future evaluation already has paid requests; do not shift its task identity.')
            episode = max(t['episode'] for t in state['tasks'] if t['phase']=='train')+1
            state['tasks'][ti:ti] = [dict(variant=k.split(':')[0], seed=int(k.split(':')[1]),
                                         phase='train', episode=episode) for k in state['learners']]
    state['torch_rng'] = torch.get_rng_state()
    store.save(name, cfg, state, status)
    return dict(policy=policy, stop_reason=state.get('training_stop_reason'),
                learners={k:dict(history=m['history'], stable_checks=m.get('stable_checks', 0))
                          for k, m in training['monitors'].items()})
