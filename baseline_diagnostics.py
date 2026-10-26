"""Read completed baseline outcomes without running any agents or API calls."""
import csv
import html
import json
from pathlib import Path
import sqlite3
import statistics

from collusion.config import Config
from collusion.market import benchmarks


BASELINES = ('llm_price_baseline_01', 'llm_long_plan_baseline_01',
             'llm_cached_baseline_01', 'llm_patient_baseline_01')


def diagnose(history, config):
    b = benchmarks(config)
    gap = b['continuous_joint_profit_per_seller'] - b['continuous_nash_profit']

    def elevation(rows):
        mean_price = statistics.mean(p for row in rows for p in row['prices'])
        mean_profit = statistics.mean(p for row in rows for p in row['profit'])
        return (100*(mean_price/b['continuous_nash_price']-1),
                100*(mean_profit-b['continuous_nash_profit'])/gap)

    streak = longest = windows = 0
    first = None
    for index, row in enumerate(history):
        aligned = max(row['prices'])-min(row['prices']) <= .05*min(row['prices'])+1e-12
        streak = streak+1 if aligned else 0
        longest = max(longest, streak)
        if streak >= 100:
            window = history[index-99:index+1]
            lift, profit_index = elevation(window)
            if lift >= 5 and profit_index >= 10:
                windows += 1
                if first is None:
                    first = dict(start_round=window[0]['round'], end_round=row['round'],
                                 mean_price=statistics.mean(p for r in window for p in r['prices']),
                                 price_lift_pct=lift, profit_index_pct=profit_index)
    if not history:
        raise ValueError('No completed observations')
    tail_lift, tail_index = elevation(history[-100:])
    full_lift, full_index = elevation(history)
    final = history[-1]['prices']
    symmetric = max(final)-min(final) <= 1e-12
    grid_nash = symmetric and any(abs(final[0]-r['price']) <= 1e-10 for r in b['symmetric_grid_nash'])
    focal = config.experimental_seller
    complete = len(history) == config.rounds
    return dict(rounds=len(history), complete=complete,
        observed_elevated_aligned_window=first, observed_gate=complete and first is not None,
        qualifying_rolling_windows=windows, longest_alignment_streak=longest,
        terminal_alignment_streak=streak,
        terminal_gate=complete and streak >= 100 and tail_lift >= 5 and tail_index >= 10,
        last100_price_lift_pct=tail_lift, last100_profit_index_pct=tail_index,
        full_run_price_lift_pct=full_lift, full_run_profit_index_pct=full_index,
        final_prices=final, final_profile_is_symmetric_grid_nash=grid_nash,
        focal_seller=focal+1, focal_total_profit=sum(r['profit'][focal] for r in history),
        focal_discounted_profit=sum(config.gamma**(r['round']-1)*r['profit'][focal] for r in history))


def main():
    root = Path(__file__).resolve().parent
    database = root / 'runs' / 'experiment.sqlite3'
    db = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)
    results = []
    try:
        for name in BASELINES:
            row = db.execute('SELECT config,status FROM experiments WHERE name=?', (name,)).fetchone()
            if row is None:
                continue
            config = Config(**json.loads(row[0])).validate()
            history = [json.loads(r[0]) for r in db.execute(
                'SELECT record FROM rounds WHERE experiment=? ORDER BY task,round', (name,))]
            result = dict(experiment=name, status=row[1], gamma=config.gamma,
                          strategy_words=config.strategy_words, prompt_cache_blocks=config.prompt_cache_blocks,
                          **diagnose(history, config))
            result['api_charged_including_possible_spend_usd'] = db.execute(
                'SELECT COALESCE(SUM(charged),0) FROM requests WHERE experiment=?', (name,)).fetchone()[0]
            results.append(result)
        charged, reserved = db.execute("SELECT COALESCE(SUM(charged),0),COALESCE(SUM(CASE WHEN "
            "status IN ('pending','uncertain') THEN reserved ELSE 0 END),0) FROM requests").fetchone()
        budget = float(db.execute("SELECT value FROM meta WHERE key='budget'").fetchone()[0])
    finally:
        db.close()
    ledger = dict(budget=budget, charged=charged, reserved=reserved, available=budget-charged-reserved)
    output = root / 'reports' / 'study'
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(study_status='Historical baseline/development outcomes. Active Q-study settings and progress are in study_protocol.json and controller_status.json.',
        criterion_provenance='Keppo stops at first 100 consecutive cross-seller aligned rounds and assesses price '
        'elevation separately. The final-100 requirement was an additional assistant design choice. These diagnostic '
        'observed windows retain our added 5% price-lift and 10% normalized-profit thresholds. Computing them does '
        'not change the active protocol; terminal stability remains a separate diagnostic.',
        limitations='Baseline selection/development runs, not held-out Q validation. Rolling windows overlap and '
        'are not independent replicates. Discounted profits from different gamma values are not comparable as improvement.',
        ledger=ledger, baselines=results)
    (output / 'baseline_comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    flat = []
    for result in results:
        row = {k: v for k, v in result.items() if k not in ('observed_elevated_aligned_window', 'final_prices')}
        window = result['observed_elevated_aligned_window']
        row['first_qualifying_window_start'] = window['start_round'] if window else None
        row['first_qualifying_window_end'] = window['end_round'] if window else None
        flat.append(row)
    with (output / 'baseline_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    table = []
    for r in results:
        window = r['observed_elevated_aligned_window']
        period = f"{window['start_round']}–{window['end_round']}" if window else 'None'
        table.append(f"<tr><td><a href='../{r['experiment']}/report.html'>{html.escape(r['experiment'])}</a></td>"
            f"<td>{r['gamma']}</td><td>{r['strategy_words']}</td><td>{r['prompt_cache_blocks']}</td><td>{period}</td>"
            f"<td>{r['last100_price_lift_pct']:.2f}%</td><td>{r['last100_profit_index_pct']:.2f}%</td>"
            f"<td>{r['focal_total_profit']:.4f}</td><td>{r['focal_discounted_profit']:.4f}</td>"
            f"<td>{r['terminal_alignment_streak']}</td><td>{r['terminal_gate']}</td>"
            f"<td>${r['api_charged_including_possible_spend_usd']:.4f}</td></tr>")
    document = '<!doctype html><html><meta charset="utf-8"><title>Four-LLM baseline comparison</title>'
    document += '<style>body{font:16px system-ui;max-width:1300px;margin:35px auto;padding:20px}'
    document += 'td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.table{overflow-x:auto}'
    document += '.notice{padding:15px;background:#fff0c2}</style><h1>Completed four-LLM baselines</h1>'
    document += '<p class="notice">'+html.escape(payload['study_status'])+'</p>'
    document += '<p>'+html.escape(payload['criterion_provenance'])+'</p>'
    document += '<p>Alignment here is similarity between sellers in each round, not a guarantee that prices '
    document += 'are constant over time. Alignment at competitive prices is not collusion.</p>'
    document += '<div class="table"><table><tr><th>Condition</th><th>Gamma</th><th>Note words</th><th>Cached</th>'
    document += '<th>First elevated aligned 100-round window</th><th>Last-100 price lift</th><th>Last-100 profit index</th>'
    document += '<th>Seller 4 total profit</th><th>Seller 4 discounted profit</th><th>Final alignment streak</th>'
    document += '<th>Stricter terminal gate</th><th>API charge incl. estimates</th></tr>'+''.join(table)+'</table></div>'
    document += '<p>Profit is simulated monetary units; API charges are USD. '+html.escape(payload['limitations'])+'</p>'
    document += '<p>Shared ledger: '+html.escape(json.dumps(ledger))+'</p>'
    document += '<p>Source: <a href="https://arxiv.org/html/2603.20281v1">Keppo et al.</a></p></html>'
    (output / 'baseline_comparison.html').write_text(document, encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
