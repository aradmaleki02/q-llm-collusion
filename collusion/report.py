import csv
import html
import json
import statistics
from pathlib import Path

from .config import Config
from .market import benchmarks


def svg_plot(series, references=(), label=""):
    width, height = 900, 260
    all_values = [v for _, values, _ in series for v in values] + [v for _, v in references]
    if not all_values:
        return ""
    lower, upper = min(all_values), max(all_values)
    padding = max((upper - lower) * 0.08, 0.01)
    lower, upper = lower - padding, upper + padding
    y = lambda value: 215 - (value - lower) / (upper - lower) * 185
    drawing = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">',
               '<rect width="900" height="260" fill="#fff"/>',
               '<path d="M55 25V215H870" fill="none" stroke="#777"/>']
    for name, value in references:
        drawing.append(f'<path d="M55 {y(value):.2f}H870" stroke="#999" stroke-dasharray="6 4"/>')
        drawing.append(f'<text x="60" y="{y(value)-4:.2f}" font-size="11">{html.escape(name)} {value:.3f}</text>')
    for name, values, color in series:
        points = " ".join(f"{55+i/max(1,len(values)-1)*815:.2f},{y(v):.2f}" for i, v in enumerate(values))
        drawing.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    drawing.append(f'<text x="5" y="30" font-size="11">{upper:.3f}</text>'
                   f'<text x="5" y="215" font-size="11">{lower:.3f}</text>'
                   '<text x="400" y="245" font-size="12">Round within this market (recorded interval)</text></svg>')
    return "".join(drawing)


def write_report(store, name, destination=None):
    values, state, status = store.load(name)
    config = Config(**values)
    focal = config.experimental_seller
    benchmark = benchmarks(config)
    destination = Path(destination or Path("reports") / name)
    destination.mkdir(parents=True, exist_ok=True)
    raw = store.records(name)
    by_task = {}
    for row in raw:
        by_task.setdefault(row["task"], []).append(json.loads(row["record"]))
    metrics, flat_rows, panels = [], [], []
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626"]
    for ti, history in by_task.items():
        task = state["tasks"][ti]
        totals = [sum(row["profit"][i] for row in history) for i in range(4)]
        discounted = sum(config.gamma ** (row["round"] - 1) * row["profit"][0] for row in history)
        interval_discounted = sum(config.gamma ** t * row["profit"][0] for t, row in enumerate(history))
        focal_discounted = sum(config.gamma ** (row["round"] - 1) * row["profit"][focal] for row in history)
        focal_interval = sum(config.gamma ** t * row["profit"][focal] for t, row in enumerate(history))
        last = history[-min(100, len(history)):]
        convergence_streak, first_convergence = 0, None
        for row in history:
            aligned = max(row['prices']) - min(row['prices']) <= .05 * min(row['prices']) + 1e-12
            convergence_streak = convergence_streak + 1 if aligned else 0
            if convergence_streak >= 100 and first_convergence is None:
                first_convergence = row['round']
        average_profit = statistics.mean(v for row in last for v in row["profit"])
        gap = benchmark["continuous_joint_profit_per_seller"] - benchmark["continuous_nash_profit"]
        index = (average_profit - benchmark["continuous_nash_profit"]) / gap if abs(gap) > 1e-12 else None
        expected = task.get("end_round", config.rounds) - len(task.get("prefix", []))
        record = dict(task=ti, variant=task["variant"], seed=task["seed"], phase=task["phase"],
                      episode=task["episode"], rounds=len(history), complete=len(history) == expected,
                      seat0_total_profit=totals[0], seat0_discounted_profit=discounted,
                      seat0_interval_discounted_profit=interval_discounted,
                      seat0_minus_mean_rival_profit=totals[0] - statistics.mean(totals[1:]),
                      mean_price_last100=statistics.mean(p for row in last for p in row["prices"]),
                      outcome_index_last100=index, backend=state["backend"])
        record.update(focal_seller=focal + 1, focal_total_profit=totals[focal],
                      full_run_price_lift_pct=100*(statistics.mean(p for row in history for p in row['prices']) /
                          benchmark['continuous_nash_price']-1),
                      full_run_profit_index_pct=100*(statistics.mean(totals)/len(history) -
                          benchmark['continuous_nash_profit'])/gap,
                      focal_discounted_profit=focal_discounted, focal_interval_discounted_profit=focal_interval,
                      focal_minus_mean_rival_profit=totals[focal] - statistics.mean(
                          totals[i] for i in range(4) if i != focal),
                      price_lift_last100_pct=100 * (record["mean_price_last100"] /
                          benchmark["continuous_nash_price"] - 1),
                      profit_collusion_index_pct=100 * index if index is not None else None,
                      price_sd_last100=statistics.pstdev(p for row in last for p in row["prices"]),
                      first_keppo_price_alignment_round=first_convergence,
                      keppo_price_alignment_at_end=convergence_streak >= 100,
                      terminal_lowest_price=min(history[-1]['prices']),
                      focal_mean_profit_last100=statistics.mean(row['profit'][focal] for row in last))
        for i in range(4):
            record[f"seller_{i}_total_profit"] = totals[i]
        metrics.append(record)
        for row in history:
            for i in range(4):
                flat_rows.append(dict(task=ti, variant=task["variant"], seed=task["seed"], phase=task["phase"],
                    episode=task["episode"], round=row["round"], seller=i, price=row["prices"][i],
                    sales=row["demand"][i], profit=row["profit"][i], loss=row["loss"]))
        title = f"{task['variant']} / seed {task['seed']} / {task['phase']} {task['episode']}"
        plot = svg_plot([(f"Seller {i}", [row["prices"][i] for row in history], colors[i]) for i in range(4)],
            [("Continuous Nash", benchmark["continuous_nash_price"]),
             ("Continuous joint optimum", benchmark["continuous_joint_price"])], title)
        panels.append(f"<h3>{html.escape(title)}</h3><p>" +
            " | ".join(f'<span style="color:{colors[i]}">Seller {i}</span>' for i in range(4)) + f"</p>{plot}")
    for filename, rows in (("metrics.csv", metrics), ("rounds.csv", flat_rows)):
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (destination / "benchmarks.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    evals = [m for m in metrics if m["phase"] == "eval" and m["complete"]]
    summary = []
    comparisons = []
    references = {(m["seed"], m["episode"]): m for m in evals if m["variant"] == "all_llm"}
    for m in evals:
        reference = references.get((m["seed"], m["episode"]))
        if m["variant"] != "all_llm" and reference:
            comparisons.append(dict(variant=m["variant"], seed=m["seed"], episode=m["episode"],
                focal_seller=focal+1, own_total_profit=m["focal_total_profit"],
                llm_seat_total_profit=reference["focal_total_profit"],
                difference=m["focal_total_profit"]-reference["focal_total_profit"],
                discounted_difference=m["focal_discounted_profit"]-reference["focal_discounted_profit"]))
    for variant in sorted({m["variant"] for m in evals}):
        records = [m for m in evals if m["variant"] == variant]
        vals = [m["focal_total_profit"] for m in records]
        summary.append(dict(variant=variant, complete_eval_markets=len(records),
            seeds=len({m["seed"] for m in records}),
            mean_seat0_profit=statistics.mean(m["seat0_total_profit"] for m in records),
            focal_seller=focal+1, mean_focal_profit=statistics.mean(vals),
            market_sd=statistics.stdev(vals) if len(vals) > 1 else None))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    controls = {(m["variant"], m["seed"], m["episode"]): m for m in metrics if m["phase"] == "probe_control"}
    gains = []
    for m in metrics:
        control = controls.get((m["variant"], m["seed"], m["episode"]))
        if m["phase"] in ("probe_undercut", "probe_increase") and control:
            gains.append(dict(branch=m["phase"], complete=m["complete"] and control["complete"],
                rounds=m["rounds"], discounted_deviation_gain=m["focal_interval_discounted_profit"]
                - control["focal_interval_discounted_profit"]))
    (destination / "probe_gains.json").write_text(json.dumps(gains, indent=2), encoding="utf-8")
    rows_html = "".join(f"<tr><td>{html.escape(m['variant'])}</td><td>{m['seed']}</td><td>{m['phase']}</td>"
        f"<td>{m['episode']}</td><td>{m['rounds']}</td><td>{m['focal_total_profit']:.5f}</td>"
        f"<td>{m['focal_minus_mean_rival_profit']:.5f}</td><td>{m['profit_collusion_index_pct']:.1f}%</td></tr>" for m in metrics)
    watermark = "OFFLINE MOCK DATA — NOT LLM EVIDENCE" if state["backend"] == "mock" else "OpenAI API experiment"
    document = f"""<!doctype html><html><meta charset="utf-8"><title>{html.escape(name)}</title>
    <style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#172033}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
    svg{{width:100%;max-height:300px}}pre{{white-space:pre-wrap}}.banner{{background:#fff0c2;padding:15px}}</style>
    <h1>{html.escape(name)}</h1><p class="banner">{watermark}</p><p>Status: {html.escape(status)}</p>
    <p>Shared ledger: {html.escape(json.dumps(store.spending()))}</p>
    <p>Outcome index measures profit elevation against continuous four-seller benchmarks. It is not a
    causal collusion detector, is not clipped to [0,1], and can be affected by the common price grid.
    Grid benchmarks are in benchmarks.json. Seller {focal+1} (zero-based index {focal}) is the focal seat,
    replaced by the Q-agent in mixed markets.
    Partial markets and training results are not held-out evaluation results.</p>
    <p>Multiple evaluation markets from one trained network are correlated. Independent training seeds,
    not rounds or calls, are the replication unit. One seed is exploratory.</p>
    <table><tr><th>Variant</th><th>Seed</th><th>Phase</th><th>Episode</th><th>Rounds</th>
    <th>Focal seller profit</th><th>Minus mean rival</th><th>Last-100 profit index</th></tr>{rows_html}</table>
    {''.join(panels)}<h2>Frozen experiment configuration</h2><pre>{html.escape(json.dumps(values,indent=2))}</pre>
    <p>Sources: <a href="https://arxiv.org/html/2603.20281v1">Keppo et al.</a>;
    <a href="https://developers.openai.com/api/docs/models/gpt-6-luna">OpenAI model/pricing</a>.</p></html>"""
    (destination / "report.html").write_text(document, encoding="utf-8")
    return destination.resolve()
