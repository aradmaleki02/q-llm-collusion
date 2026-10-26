"""Run a Q-learning study under the shared API allowance."""
import argparse
import copy
import csv
import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch

from baseline_diagnostics import diagnose
from collusion.config import Config
from collusion.convergence import advance_training
from collusion.parallel import run_parallel
from collusion.engine import make_experiment, run_experiment
from collusion.report import write_report
from collusion.learner import VARIANTS
from collusion.storage import Paused, Store, single_writer

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / 'study_protocol.json').read_text(encoding='utf-8'))
BASELINE = PROTOCOL['primary_baseline']
OUTPUT = ROOT / 'reports' / 'study'


def save_json(filename, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / filename
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(destination)


def status(stage, **details):
    value = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 process_id=os.getpid(), stage=stage, **details)
    save_json('controller_status.json', value)
    print(json.dumps(value), flush=True)


def metrics(store, name):
    with (write_report(store, name) / 'metrics.csv').open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def select_candidate(rows, external_reference=None):
    evaluated = [r for r in rows if r['phase'] == 'eval' and r['complete'] == 'True']
    reference = next((r for r in evaluated if r['variant'] == 'all_llm'), external_reference)
    ordinary = next((r for r in evaluated if r['variant'] == 'ordinary'), None)
    if reference is None or ordinary is None:
        return None, []
    comparisons = []
    for row in evaluated:
        if row['variant'] in ('ordinary', 'all_llm'):
            continue
        item = dict(variant=row['variant'], total_profit=float(row['focal_total_profit']),
                    discounted_profit=float(row['focal_discounted_profit']),
                    llm_reference_source=reference.get('reference_source', 'same_suite_reference'),
                    profit_index_pct=float(row['full_run_profit_index_pct']),
                    price_lift_pct=float(row['full_run_price_lift_pct']),
                    last100_profit_index_pct=float(row['profit_collusion_index_pct']))
        for label, control in (('llm', reference), ('ordinary', ordinary)):
            item['total_gain_vs_'+label] = item['total_profit']-float(control['focal_total_profit'])
            item['discounted_gain_vs_'+label] = item['discounted_profit']-float(control['focal_discounted_profit'])
        item['meets_discovery_target'] = all(item[k] > 0 for k in (
            'total_gain_vs_llm', 'total_gain_vs_ordinary', 'discounted_gain_vs_llm',
            'discounted_gain_vs_ordinary')) and item['profit_index_pct'] >= 10 and item['price_lift_pct'] >= 5
        comparisons.append(item)
    qualified = [r for r in comparisons if r['meets_discovery_target']]
    return (max(qualified, key=lambda r: r['discounted_gain_vs_ordinary']) if qualified else None), comparisons


def make_validation(store, source_name, variant, config, repetitions=1):
    name = PROTOCOL.get('validation_name', 'q_validation_01')
    if store.exists(name):
        return name
    _, source, source_status = store.load(source_name)
    if source_status != 'complete':
        raise Paused('Finish discovery before validation.')
    learners = {f'{v}:1': copy.deepcopy(source['learners'][f'{v}:1']) for v in ('ordinary', variant)}
    for learner in learners.values():
        learner.replay, learner.episodes = [], {}
    state = dict(backend='openai', tasks=[dict(variant=v, seed=1, phase='eval', episode=e)
        for e in range(1, repetitions+1) for v in ('ordinary', variant, 'all_llm')],
        task_index=0, market=None, learners=learners, committed_rounds=0,
        torch_rng=source['torch_rng'], version=1, study_role='fresh frozen-policy validation', source=source_name)
    store.save(name, config, state, 'ready')
    return name


def cap_stage(store, name, allowance):
    values, state, previous_status = store.load(name)
    if allowance is None:
        if state.get('spend_ceiling') is not None or state.get('stage_allowance_usd') is not None:
            state.setdefault('budget_amendments', []).append(dict(
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                previous_ceiling=state.get('spend_ceiling'),
                previous_stage_allowance=state.get('stage_allowance_usd'),
                reason=PROTOCOL.get('stage_cap_authorization', 'Stage allowance disabled in protocol.')))
        ledger = store.spending()
        state.setdefault('stage_start_used', ledger['charged']+ledger['reserved'])
        state.update(spend_ceiling=None, stage_allowance_usd=None)
        store.save(name, Config(**values), state, previous_status)
        return None
    if 'spend_ceiling' not in state:
        ledger = store.spending()
        used = ledger['charged']+ledger['reserved']
        state.update(spend_ceiling=min(ledger['budget'], used+allowance),
                     stage_start_used=used, stage_allowance_usd=allowance)
        store.save(name, Config(**values), state, previous_status)
    return state['spend_ceiling']


def warm_start(config, paths, require_ready=True):
    learners, provenance = {}, {}
    for variant, relative in paths.items():
        path = ROOT / relative
        saved = torch.load(path, map_location='cpu', weights_only=False)
        learner, report = saved['learner'], saved['report']
        compatible = replace(learner.config, device=config.device, epsilon_start=config.epsilon_start,
                             epsilon_end=config.epsilon_end, epsilon_decay_steps=config.epsilon_decay_steps)
        if ((require_ready and not report['ready']) or compatible != config
                or learner.kappa != VARIANTS[variant]):
            raise Paused(f'Local {variant} policy is not ready or has mismatched settings.')
        if not all(torch.isfinite(p).all() for p in learner.online.parameters()):
            raise Paused(f'Non-finite {variant} initialization.')
        provenance[variant] = dict(file=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   local_steps=learner.steps, local_updates=learner.updates,
                                   local_controls_passed=report['ready'],
                                   readiness_results=report['results'])
        learner.config = config
        learner.steps, learner.updates = 0, 0
        learner.replay, learner.episodes = [], {}
        learner.target.load_state_dict(learner.online.state_dict())
        learner.optimizer = torch.optim.Adam(learner.online.parameters(), lr=config.learning_rate)
        learners[f'{variant}:1'] = learner.to(config.device)
    return learners, provenance


def run_stage(store, name, adaptive_policy=None):
    _, checkpoint, _ = store.load(name)
    convergence = None
    def progress(message, markets=None):
        print(message, flush=True)
        ledger = store.spending()
        status('running', experiment=name, progress=message, spending=ledger,
               stage_used_usd=ledger['charged']+ledger['reserved']-checkpoint['stage_start_used'],
               stage_allowance_usd=checkpoint['stage_allowance_usd'],
               absolute_spend_ceiling=checkpoint['spend_ceiling'],
               completed_training_games={k:len(v['history']) for k,v in convergence['learners'].items()}
                   if convergence else None, markets=markets)
        number = message.partition('round=')[2].partition('/')[0]
        if number.isdigit() and (int(number) == 1 or int(number) % 100 == 0):
            write_report(store, name)
    def publish_convergence(value):
        nonlocal convergence
        convergence = value
        save_json('training_convergence.json', value)
    if adaptive_policy and PROTOCOL.get('parallel_markets'):
        result = run_parallel(store, name, adaptive_policy, PROTOCOL['learner_devices'],
                              progress=progress, on_convergence=publish_convergence)
        rows = metrics(store, name)
        if result != 'complete':
            raise Paused(result)
        return rows
    while True:
        if adaptive_policy:
            convergence = advance_training(store, name, adaptive_policy)
            save_json('training_convergence.json', convergence)
        result = run_experiment(store, name, progress=progress, stop_after_task=bool(adaptive_policy))
        if result == 'task_complete':
            metrics(store, name)
            continue
        rows = metrics(store, name)
        if result != 'complete':
            raise Paused(result)
        return rows


def main(validation_variant=None):
    os.chdir(ROOT)
    torch.set_num_threads(1)
    with single_writer(ROOT / 'runs'):
        store = Store(ROOT / 'runs')
        try:
            if PROTOCOL.get('local_capacity_audit_pending'):
                raise Paused('Local capacity/readiness checks are pending; no paid training will start.')
            if store.budget() > PROTOCOL['maximum_shared_api_allowance_usd']:
                raise Paused('Shared allowance exceeds the authorized study limit.')
            values, _, baseline_status = store.load(BASELINE)
            baseline_config = Config(**values).validate()
            history = [json.loads(r['record']) for r in store.records(BASELINE)]
            evidence = diagnose(history, baseline_config)
            if baseline_status != 'complete' or not evidence['observed_gate']:
                raise Paused('The saved baseline has no qualifying elevated, aligned 100-round window.')
            save_json('baseline_observation.json', dict(experiment=BASELINE, **evidence))
            reference = dict(metrics(store, BASELINE)[0], reference_source=BASELINE)
            name = PROTOCOL['discovery_name']
            config = replace(baseline_config, **{k: PROTOCOL[k] for k in (
                'updates_per_step', 'hidden_size', 'head_hidden_size', 'device', 'epsilon_start',
                'epsilon_end', 'epsilon_decay_steps') if k in PROTOCOL})
            if validation_variant:
                selected, _ = select_candidate(metrics(store, name), reference)
                if not selected or selected['variant'] != validation_variant:
                    raise Paused('No qualifying development candidate; review before paying for validation.')
                name = make_validation(store, name, validation_variant, config)
                allowance = PROTOCOL['validation_allowance_usd']
            else:
                allowance = PROTOCOL['discovery_allowance_usd']
                if not store.exists(name):
                    if store.spending()['available'] < (allowance or 0)+PROTOCOL['validation_reserve_usd']:
                        raise Paused('Discovery would consume the reserved validation funds.')
                    initial, provenance = warm_start(config, PROTOCOL['initial_policy_files'],
                        PROTOCOL.get('require_local_oracle_gate', True)) if PROTOCOL.get(
                        'initial_policy_files') else (None, {})
                    make_experiment(store, name, config, 'openai', ['ordinary', 'optimistic'], [1], 1, 1,
                                    include_llm_reference=False)
                    _, state, previous_status = store.load(name)
                    state.update(study_role='development only', external_llm_reference=BASELINE,
                                 prospective_protocol=copy.deepcopy(PROTOCOL), local_pretraining=provenance)
                    if initial:
                        state['learners'] = initial
                    if PROTOCOL.get('supersedes_discovery'):
                        old_name = PROTOCOL['supersedes_discovery']
                        old_config, old_state, _ = store.load(old_name)
                        for key in ('spend_ceiling', 'stage_start_used', 'stage_allowance_usd'):
                            state[key] = old_state[key]
                        old_state.update(excluded_from_primary_analysis=True, superseded_by=name)
                        store.save(old_name, Config(**old_config), old_state,
                                   'archived: replaced after human capacity audit')
                    store.save(name, config, state, previous_status)
            saved_config, checkpoint, _ = store.load(name)
            if not validation_variant and PROTOCOL.get('initial_policy_files') and not checkpoint.get('local_pretraining'):
                raise Paused('Pretrained policy provenance is missing; no paid training will start.')
            if Config(**saved_config) != config:
                raise Paused('Existing experiment settings differ; choose a new name instead of changing a saved study.')
            ceiling = cap_stage(store, name, allowance)
            status('starting', experiment=name, allowance_usd=allowance,
                   absolute_spend_ceiling=ceiling, spending=store.spending())
            rows = run_stage(store, name, None if validation_variant else PROTOCOL.get('training_convergence'))
            selected, comparisons = select_candidate(rows, reference)
            save_json('validation_comparisons.json' if validation_variant else 'discovery_comparisons.json', comparisons)
            _, completed, _ = store.load(name)
            stop_reason = completed.get('training_stop_reason')
            final_status = ('validation_complete' if validation_variant else
                            'development_evaluation_complete_budget_limited_not_converged' if
                            stop_reason == 'budget_limited_not_converged' else 'discovery_complete_needs_review')
            status(final_status, experiment=name, selected=selected, comparisons=comparisons,
                   training_stop_reason=stop_reason, spending=store.spending())
        finally:
            store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', choices=['optimistic', 'strong'])
    args = parser.parse_args()
    try:
        main(args.validate)
    except Paused as exc:
        status('paused_needs_attention', reason=str(exc))
