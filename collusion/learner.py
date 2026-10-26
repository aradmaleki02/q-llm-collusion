import copy
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn import functional as F


VARIANTS = {"ordinary": 1.0, "optimistic": 0.25, "strong": 0.10}


def observations(history, config):
    """Every historical observation, including a start token; no window truncation.

    Only the experimental seller's demand/profit is visible; no private rival plans.
    Remaining-time information is present only for the known finite gamma=1 game.
    """
    span = config.price_max - config.price_min
    remaining = lambda t: (config.rounds - t) / config.rounds if config.gamma == 1 else 0.0
    rows = [[0.0] * 6 + [remaining(0)]]
    for t, row in enumerate(history, 1):
        rows.append([(p - config.price_min) / span for p in row["prices"]]
                    + [row["demand"][config.experimental_seller], row["profit"][config.experimental_seller], remaining(t)])
    return torch.tensor(rows, dtype=torch.float32)


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(7, config.hidden_size, batch_first=True)
        self.skip_latest = config.head_hidden_size > 0
        self.head = (nn.Sequential(nn.Linear(config.hidden_size+7, config.head_hidden_size), nn.ReLU(),
                                  nn.Linear(config.head_hidden_size, config.price_count))
                     if self.skip_latest else nn.Linear(config.hidden_size, config.price_count))

    def forward(self, sequences):
        lengths = [len(s) for s in sequences]
        device = next(self.parameters()).device
        padded = pad_sequence(sequences, batch_first=True).to(device)
        if device.type == "cuda":
            # Gather the last REAL timestep; padding must not alter the state.
            output, _ = self.gru(padded)
            hidden = output[torch.arange(len(lengths), device=device),
                            torch.tensor(lengths, device=device)-1]
        else:
            packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
            _, recurrent = self.gru(packed)
            hidden = recurrent[-1]
        if getattr(self, 'skip_latest', False):
            latest = padded[torch.arange(len(lengths), device=device),
                            torch.tensor(lengths, device=device)-1]
            hidden = torch.cat((hidden, latest), dim=1)
        return self.head(hidden)


def td_targets(reward, next_value, terminal, gamma):
    return reward + gamma * next_value * (~terminal).float()


def hysteretic_loss(predicted, targets, kappa):
    weight = torch.where(targets.detach() >= predicted.detach(), 1.0, kappa)
    return (weight * F.smooth_l1_loss(predicted, targets.detach(), reduction="none")).mean()


class Learner:
    def __init__(self, config, kappa, seed):
        self.config, self.kappa = config, kappa
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = Network(config)
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.steps = 0
        self.updates = 0
        # Replay indexes complete prefixes in episode records instead of copying
        # growing histories into every transition. Prefixes are re-encoded with
        # current weights; stored recurrent hidden states never become stale.
        self.episodes = {}
        self.replay = []
        self.to(config.device)

    def to(self, device):
        self.online.to(device)
        self.target.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        return self

    def act(self, history, training):
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * max(
            0.0, 1 - self.steps / self.config.epsilon_decay_steps)
        if training and self.rng.random() < epsilon:
            return self.rng.randrange(self.config.price_count)
        with torch.no_grad():
            return int(self.online([observations(history, self.config)])[0].argmax())

    def learn(self, episode_key, history, action):
        self.steps += 1
        self.episodes[episode_key] = history
        self.replay.append((episode_key, len(history) - 1, action))
        self.replay = self.replay[-self.config.replay_capacity:]
        live = {key for key, _, _ in self.replay}
        self.episodes = {key: value for key, value in self.episodes.items() if key in live}
        if len(self.replay) < self.config.warmup_transitions:
            return None
        return sum(self._update() for _ in range(self.config.updates_per_step)) / self.config.updates_per_step

    def _update(self):
        """One replay update; additional updates require no market/API calls."""
        batch = self.rng.sample(self.replay, min(self.config.batch_size, len(self.replay)))
        before, after, rewards, actions, terminals = [], [], [], [], []
        for key, t, action_index in batch:
            rows = self.episodes[key]
            before.append(observations(rows[:t], self.config))
            after.append(observations(rows[:t + 1], self.config))
            actions.append(action_index)
            rewards.append(rows[t]["profit"][self.config.experimental_seller])
            # A gamma=.95 market is a continuing game truncated at the cap.
            # Only a known finite gamma=1 endpoint zeros the bootstrap.
            terminals.append(self.config.gamma == 1 and t + 1 == self.config.rounds)
        device = next(self.online.parameters()).device
        predicted = self.online(before).gather(1, torch.tensor(actions, device=device)[:, None]).squeeze(1)
        with torch.no_grad():
            best_actions = self.online(after).argmax(1)
            next_value = self.target(after).gather(1, best_actions[:, None]).squeeze(1)
            targets = td_targets(torch.tensor(rewards, device=device), next_value,
                                 torch.tensor(terminals, device=device), self.config.gamma)
        loss = hysteretic_loss(predicted, targets, self.kappa)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; previous committed round remains recoverable")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
