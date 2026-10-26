from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str = "gpt-6-luna"
    reasoning_effort: str = "low"
    max_output_tokens: int = 4096
    timeout_seconds: int = 120
    rounds: int = 1000
    gamma: float = 0.95
    experimental_seller: int = 0
    parallel_llm_calls: int = 1
    action_interface: str = "index"
    strategy_words: int = 40
    prompt_cache_blocks: bool = False
    quality: float = 2.0
    outside_quality: float = 0.0
    mu: float = 0.25
    marginal_cost: float = 1.0
    price_min: float = 1.0
    price_max: float = 3.0
    price_count: int = 51
    hidden_size: int = 32
    head_hidden_size: int = 0
    device: str = "cpu"
    learning_rate: float = 0.0003
    updates_per_step: int = 1
    batch_size: int = 8
    replay_capacity: int = 5000
    warmup_transitions: int = 64
    target_update_interval: int = 100
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    input_per_million: float = 0.10
    cached_input_per_million: float = 0.01
    cache_write_per_million: float = 0.125
    output_per_million: float = 0.50

    def validate(self):
        if self.device not in ("cpu", "cuda:0", "cuda:1"):
            raise ValueError("device must be cpu, cuda:0 or cuda:1")
        if type(self.head_hidden_size) is not int or self.head_hidden_size < 0:
            raise ValueError("head_hidden_size must be a nonnegative integer")
        if not all(not isinstance(v, float) or math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration numbers must be finite.")
        for key in ("rounds", "hidden_size", "batch_size", "replay_capacity", "warmup_transitions",
                    "target_update_interval", "epsilon_decay_steps", "max_output_tokens", "timeout_seconds",
                    "parallel_llm_calls", "updates_per_step"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.experimental_seller) is not int or not 0 <= self.experimental_seller < 4:
            raise ValueError("experimental_seller must be 0, 1, 2 or 3")
        if self.parallel_llm_calls > 4:
            raise ValueError("At most four simultaneous seller requests are needed")
        if self.action_interface not in ("index", "price"):
            raise ValueError("action_interface must be index or price")
        if type(self.strategy_words) is not int or not 1 <= self.strategy_words <= 1000:
            raise ValueError("strategy_words must be an integer from 1 to 1000")
        if type(self.prompt_cache_blocks) is not bool:
            raise ValueError("prompt_cache_blocks must be true or false")
        if not isinstance(self.price_count, int) or self.price_count < 2:
            raise ValueError("price_count must be at least two")
        if not (0 < self.gamma <= 1 and self.mu > 0 and self.learning_rate > 0):
            raise ValueError("Invalid gamma, mu or learning rate")
        if not (self.price_min >= self.marginal_cost and self.price_max > self.price_min):
            raise ValueError("Require price_max > price_min >= marginal_cost")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if min(self.input_per_million, self.cached_input_per_million,
               self.cache_write_per_million, self.output_per_million) < 0:
            raise ValueError("Token prices cannot be negative")
        if not self.model or self.reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Invalid model or reasoning effort")
        return self

    @property
    def prices(self):
        return [round(self.price_min + i * (self.price_max - self.price_min) / (self.price_count - 1), 10)
                for i in range(self.price_count)]

    def dictionary(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values).validate()
