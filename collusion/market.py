import math


def outcome(prices, config):
    """Stable multinomial logit demand; market mass is one, not inventory."""
    utilities = [(config.quality - p) / config.mu for p in prices]
    outside = config.outside_quality / config.mu
    offset = max(outside, *utilities)
    weights = [math.exp(u - offset) for u in utilities]
    denominator = math.exp(outside - offset) + sum(weights)
    demand = [w / denominator for w in weights]
    profit = [(p - config.marginal_cost) * q for p, q in zip(prices, demand)]
    return demand, profit


def _root(function, lower, upper):
    if function(lower) * function(upper) > 0:
        raise ValueError("Benchmark root is not bracketed")
    for _ in range(100):
        middle = (lower + upper) / 2
        if function(middle) > 0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2


def benchmarks(config):
    n = 4
    c, mu = config.marginal_cost, config.mu
    share = lambda p: outcome([p] * n, config)[0][0]
    competitive = _root(lambda p: p - c - mu / (1 - share(p)), c, config.quality + c + 20 * mu)
    monopoly = _root(lambda p: p - c - mu * (1 + n * math.exp(
        (config.quality - p - config.outside_quality) / mu)), c, config.quality + c + 20 * mu)
    grid_nash = []
    for index, p in enumerate(config.prices):
        payoff = outcome([p] * n, config)[1][0]
        best = max(outcome([candidate, p, p, p], config)[1][0] for candidate in config.prices)
        if payoff >= best - 1e-12:
            grid_nash.append({"price": p, "profit": payoff, "index": index})
    p_joint = max(config.prices, key=lambda p: sum(outcome([p] * n, config)[1]))
    return {
        "continuous_nash_price": competitive,
        "continuous_nash_profit": outcome([competitive] * n, config)[1][0],
        "continuous_joint_price": monopoly,
        "continuous_joint_profit_per_seller": outcome([monopoly] * n, config)[1][0],
        "symmetric_grid_nash": grid_nash,
        "symmetric_grid_joint_price": p_joint,
        "symmetric_grid_joint_profit_per_seller": outcome([p_joint] * n, config)[1][0],
    }
