"""Two independent markets, one checkpoint owner and one atomic shared API ledger."""
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
import json
import time

import torch

from .api import Gateway
from .config import Config
from .convergence import advance_training, record_game, training_stop_reason
from .market import outcome
from .storage import Paused, Store


def add_training_tasks(state, target):
    existing = {(t['variant'], t['seed'], t['episode']) for t in state['tasks'] if t['phase']=='train'}
    for key in state['learners']:
        variant, seed = key.split(':')
        for episode in range(1, target+1):
            if (variant, int(seed), episode) not in existing:
                # Append only. Historical task IDs, paid requests and replay keys never change.
                state['tasks'].append(dict(variant=variant, seed=int(seed), phase='train', episode=episode))


def initialize_parallel(state, devices):
    if state.get('parallel_execution'):
        if state['parallel_execution']['devices'] != devices:
            raise Paused('Saved parallel device assignments differ; review before changing them.')
        return
    completed = list(range(state['task_index']))
    market = state.get('market')
    active = {state['task_index']: market} if market is not None else {}
    counts = [max((t['episode'] for i,t in enumerate(state['tasks']) if i in completed
                  and t['phase']=='train' and f"{t['variant']}:{t['seed']}"==key), default=0)
              for key in state['learners']]
    pending = [t['episode'] for i,t in enumerate(state['tasks']) if i not in completed and t['phase']=='train']
    target = max(max(counts)+(max(counts)!=min(counts)), max(pending, default=1))
    state['parallel_execution'] = dict(devices=devices, completed_tasks=completed, markets=active,
        cycle_target=target, phase='evaluation' if state.get('training_stop_reason') else 'training',
        migrated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        migration_rounds=state['committed_rounds'], historical_task_ids_preserved=True,
        migration_learner_progress={k:dict(steps=l.steps, gradient_updates=l.updates,
            replay_transitions=len(l.replay)) for k,l in state['learners'].items()})
    state['market'] = None
    if not state.get('training_stop_reason'):
        add_training_tasks(state, target)


def run_parallel(store, name, policy, devices, progress=lambda *_: None, on_convergence=lambda _: None):
    values, state, saved_status = store.load(name)
    if saved_status.startswith('archived') or state.get('excluded_from_primary_analysis'):
        raise Paused('Archived experiments cannot resume.')
    if 'training_convergence' not in state:
        advance_training(store, name, policy)
        values, state, _ = store.load(name)
    if state['training_convergence']['policy'] != policy:
        raise Paused('Saved convergence criteria differ from the current protocol.')
    cfg = Config(**values).validate()
    initialize_parallel(state, devices)
    execution = state['parallel_execution']
    torch.set_num_threads(1)
    torch.set_rng_state(state['torch_rng'])
    for key, learner in state['learners'].items():
        learner.config = replace(cfg, device=devices[key.split(':')[0]])
        learner.to(learner.config.device)
    store.recover_inflight(name)
    unresolved = store.db.execute("SELECT id,status FROM requests WHERE experiment=? AND status IN "
                                  "('uncertain','invalid') ORDER BY id LIMIT 1", (name,)).fetchone()
    if unresolved:
        raise Paused(f"Request {unresolved['id']} is {unresolved['status']}; inspect before resuming. No replacement was sent.")

    def save(status='running', row=None):
        state['torch_rng'] = torch.get_rng_state()
        incomplete = [i for i in range(len(state['tasks'])) if i not in execution['completed_tasks']]
        state['task_index'] = min(execution['markets'], default=min(incomplete, default=len(state['tasks'])))
        store.save(name, cfg, state, status, row)

    def summary():
        return {state['tasks'][ti]['variant']:dict(task=ti, phase=state['tasks'][ti]['phase'],
            game=state['tasks'][ti]['episode'], completed_rounds=len(m['history']),
            device=devices[state['tasks'][ti]['variant']]) for ti,m in execution['markets'].items()}

    def convergence_summary():
        monitors = state['training_convergence']['monitors']
        on_convergence(dict(policy=policy, stop_reason=state.get('training_stop_reason'),
            learners={k:dict(history=m['history'], stable_checks=m.get('stable_checks',0)) for k,m in monitors.items()}))

    def finish_market(ti):
        task, market = state['tasks'][ti], execution['markets'][ti]
        if task['phase']=='train' and ti not in state['training_convergence']['recorded_tasks']:
            key = f"{task['variant']}:{task['seed']}"
            monitor = state['training_convergence']['monitors'].setdefault(key, dict(history=[],probes=[]))
            cost = store.db.execute('SELECT COALESCE(SUM(charged),0) FROM requests WHERE experiment=? '
                                    'AND position LIKE ?', (name,f'{ti}/%')).fetchone()[0]
            record_game(state['learners'][key], market['history'], task, policy, monitor, cost)
            state['training_convergence']['recorded_tasks'].append(ti)
        execution['completed_tasks'].append(ti)
        del execution['markets'][ti]
        save()
        convergence_summary()

    def decide(ti, seller, market):
        worker = Store(store.root)
        try:
            gateway = Gateway(worker, cfg, name, state['backend'], spend_ceiling=state.get('spend_ceiling'))
            return gateway.decide(f"{ti}/{len(market['history'])+1}/{seller}", seller,
                                  market['history'], market['strategies'][seller])
        finally:
            worker.close()

    def commit_round(ti, decisions):
        task, market = state['tasks'][ti], execution['markets'][ti]
        learner = state['learners'][f"{task['variant']}:{task['seed']}"]
        actions, strategies = [None]*4, market['strategies'].copy()
        actions[cfg.experimental_seller] = market['pending']['q_action']
        for seller, decision in decisions.items():
            actions[seller], strategies[seller] = decision['index'], decision['strategy']
        prices = [cfg.prices[a] for a in actions]
        demand, profit = outcome(prices,cfg)
        record = dict(round=len(market['history'])+1, prices=prices, actions=actions,
                      demand=demand, profit=profit, strategies=strategies, loss=None)
        market['history'].append(record)
        if task['phase']=='train':
            record['loss'] = learner.learn(f'task:{ti}',market['history'],actions[cfg.experimental_seller])
        market['strategies'], market['pending'] = strategies, None
        state['committed_rounds'] += 1
        save(row=(ti,task,record))
        if record['round']==1 or record['round']%25==0 or record['round']==cfg.rounds:
            progress(f"{name} | {task['variant']} {task['phase']} game={task['episode']} "
                     f"round={record['round']}/{cfg.rounds} gradient_updates={learner.updates} | "
                     f"spent=${store.spending()['charged']:.3f}/$50.00", summary())

    jobs, futures, halt = {}, {}, None
    save()
    convergence_summary()
    progress('Parallel markets started; separate histories, three concurrent API calls per market.', summary())
    with ThreadPoolExecutor(max_workers=6) as pool:
        try:
            while True:
                if (store.root/'pause_requested').exists():
                    halt = halt or 'Pause requested; active API calls completed normally.'
                for ti, market in list(execution['markets'].items()):
                    if ti not in jobs and len(market['history'])>=cfg.rounds:
                        finish_market(ti)
                if not halt and not execution['markets'] and not jobs:
                    cycle_done = all(i in execution['completed_tasks'] for i,t in enumerate(state['tasks'])
                                     if t['phase']=='train' and t['episode']<=execution['cycle_target'])
                    if execution['phase']=='training' and cycle_done:
                        state['training_stop_reason'] = training_stop_reason(state,store.spending()['available'],policy)
                        if state['training_stop_reason']:
                            execution['phase'] = 'evaluation'
                        else:
                            execution['cycle_target'] += 1
                            add_training_tasks(state,execution['cycle_target'])
                        save()
                        convergence_summary()
                    elif execution['phase']=='evaluation' and all(i in execution['completed_tasks']
                            for i,t in enumerate(state['tasks']) if t['phase']=='eval'):
                        execution['phase'] = 'complete'
                        save('complete')
                        return 'complete'
                if not halt:
                    active_variants = {state['tasks'][ti]['variant'] for ti in execution['markets']}
                    phase = 'train' if execution['phase']=='training' else 'eval'
                    for key in state['learners']:
                        variant = key.split(':')[0]
                        if variant in active_variants:
                            continue
                        candidates = [i for i,t in enumerate(state['tasks']) if i not in execution['completed_tasks']
                            and t['phase']==phase and t['variant']==variant and
                            (phase=='eval' or t['episode']<=execution['cycle_target'])]
                        if candidates:
                            ti = min(candidates,key=lambda i:state['tasks'][i]['episode'])
                            task = state['tasks'][ti]
                            execution['markets'][ti] = dict(history=copy.deepcopy(task.get('prefix',[])),
                                strategies=task.get('initial_strategies',['']*4).copy(),pending=None)
                    ready = [ti for ti in execution['markets'] if ti not in jobs]
                    for ti in ready:
                        task, market = state['tasks'][ti], execution['markets'][ti]
                        if market['pending'] is None:
                            learner = state['learners'][f"{task['variant']}:{task['seed']}"]
                            market['pending'] = dict(q_action=learner.act(market['history'],task['phase']=='train'))
                    if ready:
                        save()  # Both selected actions/RNGs are durable before any paid call.
                    for ti in ready:
                        market = execution['markets'][ti]
                        jobs[ti] = dict(remaining=3,decisions={},failed=False)
                        for seller in range(4):
                            if seller!=cfg.experimental_seller:
                                future = pool.submit(decide,ti,seller,market)
                                futures[future] = (ti,seller)
                if not futures:
                    if halt:
                        save('paused: '+halt)
                        progress('PAUSED: '+halt,summary())
                        return 'paused: '+halt
                    raise Paused('Parallel scheduler has no runnable market; inspect saved task state.')
                done, _ = wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    ti,seller = futures.pop(future)
                    job = jobs[ti]
                    try:
                        job['decisions'][seller] = future.result()
                    except Exception as exc:
                        halt = halt or (str(exc) if isinstance(exc,Paused) else f'Worker error: {type(exc).__name__}')
                        job['failed'] = True
                    job['remaining'] -= 1
                    if not job['remaining']:
                        del jobs[ti]
                        if not job['failed']:
                            commit_round(ti,job['decisions'])
        except (Exception,KeyboardInterrupt) as exc:
            # Pool shutdown waits for all active requests; saved raw responses are reused.
            _, durable, _ = store.load(name)
            reason = (str(exc) if isinstance(exc,Paused) else 'Interrupted; saved state is resumable.'
                      if isinstance(exc,KeyboardInterrupt) else f'Coordinator error: {type(exc).__name__}')
            store.save(name,cfg,durable,'paused: '+reason)
            state, execution = durable, durable['parallel_execution']
            progress('PAUSED: '+reason,summary())
            return 'paused: '+reason
