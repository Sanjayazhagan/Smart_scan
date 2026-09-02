"""Double Deep Q-Network (Double DQN) Scheduler for Cognitive Radio Scanning.

Features:
- Decoupled target Q-network to eliminate overestimation bias under partial observability.
- Experience replay buffer with online Bellman gradient steps.
- Ingests observable 60-dimensional state representation.
"""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Deep Q-Network mapping 60-dim observable state to 20 action Q-values."""

    def __init__(self, state_dim: int = 60, action_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNScheduler:
    """Double DQN Scheduler with online replay memory and target decoupling."""

    def __init__(
        self,
        num_bands: int = 20,
        gamma: float = 0.92,
        lr: float = 1e-3,
        batch_size: int = 32,
        buffer_capacity: int = 2000,
        epsilon_start: float = 0.35,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_tau: float = 0.05,
        seed: int = 42,
    ):
        self.num_bands = int(num_bands)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(target_update_tau)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.state_dim = self.num_bands * 3  # 60 dims
        self.q_online = QNetwork(self.state_dim, self.num_bands)
        self.q_target = QNetwork(self.state_dim, self.num_bands)
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_capacity)

        # Internal state tracking
        self.band_belief = np.full(self.num_bands, 0.05, dtype=np.float32)
        self.scan_age = np.zeros(self.num_bands, dtype=np.float32)
        self.uncertainty = np.ones(self.num_bands, dtype=np.float32)
        self.last_state = self._get_state()
        self.last_action = 0

    def _get_state(self) -> np.ndarray:
        """Constructs normalized 60-dimensional observable feature vector."""
        norm_age = np.clip(self.scan_age / 50.0, 0.0, 1.0)
        return np.concatenate([self.band_belief, norm_age, self.uncertainty]).astype(np.float32)

    def select_band(self) -> int:
        """Epsilon-greedy action selection using online Q-network."""
        self.last_state = self._get_state()

        if self.rng.random() < self.epsilon:
            action = self.rng.randrange(self.num_bands)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(self.last_state, dtype=torch.float32).unsqueeze(0)
                q_vals = self.q_online(state_tensor).squeeze(0).numpy()
                action = int(np.argmax(q_vals))

        self.last_action = action
        return action

    def update(self, band: int, reward: float, obs_dict: dict | None = None):
        """Stores experience in replay buffer, updates Q-network, and soft-updates target."""
        band = int(band)
        self.scan_age += 1.0
        self.scan_age[band] = 0.0

        detected = 0
        if obs_dict is not None:
            detected = int(obs_dict.get("detected", 0))

        # Update beliefs
        if detected:
            self.band_belief[band] = 0.90
            self.uncertainty[band] = 0.1
        else:
            self.band_belief[band] = 0.02
            self.uncertainty[band] = 0.2

        # Uncertainty grows for unobserved bands
        for b in range(self.num_bands):
            if b != band:
                self.uncertainty[b] = min(1.0, self.uncertainty[b] + 0.04)

        next_state = self._get_state()

        # Store transition in replay buffer
        self.replay_buffer.append((
            self.last_state,
            band,
            float(reward),
            next_state,
            False,
        ))

        # Train on minibatch if buffer has enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _train_step(self):
        """Double Q-learning parameter update step."""
        batch = self.rng.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        a_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        r_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_s_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Current Q estimates
        current_q = self.q_online(s_tensor).gather(1, a_tensor)

        # Double DQN Target:
        # 1. Action selection using online network
        with torch.no_grad():
            best_next_actions = self.q_online(next_s_tensor).argmax(dim=1, keepdim=True)
            # 2. Action evaluation using target network
            target_q_next = self.q_target(next_s_tensor).gather(1, best_next_actions)
            target_q = r_tensor + self.gamma * target_q_next

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network: theta_target = tau * theta_online + (1 - tau) * theta_target
        for p_online, p_target in zip(self.q_online.parameters(), self.q_target.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1.0 - self.tau) * p_target.data)
