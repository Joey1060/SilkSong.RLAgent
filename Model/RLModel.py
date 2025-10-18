import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchrl.modules import NoisyLinear, reset_noise

import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # tree has 2*capacity elements (index 1..2*capacity-1 used)
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, data):
        # leaf index in tree array
        idx = self.ptr + self.capacity
        self.data[self.ptr] = data
        self.update(idx, priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # propagate change up to root
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, s):
        """
        Find sample on the tree given cumulative sum s
        """
        idx = 1  # start at root
        while idx < self.capacity:  # while not at leaf
            left = 2 * idx
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[1]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, beta_increment=1e-6, device="cpu"):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        self.eps = 1e-6  # small constant to avoid zero priority

    def push(self, state, action, reward, next_state, done):
        # initial priority = max priority so new samples are likely to be seen
        max_prio = np.max(self.tree.tree[-self.tree.capacity:])
        if max_prio == 0:
            max_prio = 1.0
        data = (state, action, reward, next_state, done)
        self.tree.add(max_prio, data)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_prob = np.array(priorities) / self.tree.total
        weights = (self.tree.size * sampling_prob) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.as_tensor(actions, dtype=torch.long, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.float32, device=self.device),
            torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            idxs
        )

    def update_priorities(self, idxs, td_errors):
        # priorities = |TD error|^alpha
        for idx, err in zip(idxs, td_errors.detach().cpu().numpy()):
            p = (abs(err) + self.eps) ** self.alpha
            self.tree.update(idx, p)
        
    @property
    def total(self):
        return self.tree.total


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        # Shared body
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.body = nn.Sequential(*layers)

        self.value_noisy = NoisyLinear(last, 1)
        self.action_noisy = NoisyLinear(last, action_dim)

        # Value stream: outputs scalar V(s)
        self.value = nn.Sequential(
            nn.Linear(last, last),
            nn.ReLU(),
            self.value_noisy
        )

        # Advantage stream: outputs vector A(s,·)
        self.advantage = nn.Sequential(
            nn.Linear(last, last),
            nn.ReLU(),
            self.action_noisy
        )

    def forward(self, state):
        features = self.body(state)            # [B, hidden]
        v = self.value(features)               # [B, 1]
        a = self.advantage(features)           # [B, action_dim]

        a_mean = a.mean(dim=1, keepdim=True)   # [B, 1]
        q = v + (a - a_mean)                   # [B, action_dim]
        return q
    
    def reset_noise(self):
        reset_noise(self.value_noisy)
        reset_noise(self.action_noisy)

class ReducedBinaryActionSpace:
    def __init__(self, n_bits, constraints=None):
        all_actions = ((torch.arange(2**n_bits)[:, None] >> torch.arange(n_bits)) & 1).int()
        if constraints is not None:
            mask = constraints(all_actions)
            self.valid_actions = all_actions[mask]
        else:
            self.valid_actions = all_actions

        self.n_bits = n_bits
        self.n = len(self.valid_actions)

        self.powers = 2 ** torch.arange(n_bits)
        keys = (self.valid_actions * self.powers).sum(dim=1).tolist()
        self.int_to_idx = {k: i for i, k in enumerate(keys)}
        self.idx_to_action = self.valid_actions  # keep as tensor

    def sample(self, batch_size=1):
        return torch.randint(0, self.n, (batch_size,))

    def index_to_action(self, idx):
        """
        idx: int or tensor of shape [B]
        returns: [n_bits] or [B, n_bits]
        """
        
        return self.idx_to_action[idx]

    def action_to_index(self, action_bits):
        """
        action_bits: [n_bits] or [B, n_bits]
        returns: int or tensor of shape [B]
        """
        if action_bits.ndim == 1:
            key = int((action_bits * self.powers).sum().item())
            return self.int_to_idx[key]
        else:
            keys = (action_bits * self.powers).sum(dim=1).tolist()
            return torch.tensor([self.int_to_idx[k] for k in keys], dtype=torch.long)

    def __len__(self):
        return self.n
    

class DQNAgent:
    def __init__(
        self,
        state_dim,
        num_valid_actions,
        dueling_net_cls,           # e.g. DuelingQNetwork
        hidden_sizes=(32,64),
        lr=3e-4,
        gamma=0.99,
        tau=1.0,                   # hard update when tau=1.0; soft if <1
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.q_net = dueling_net_cls(state_dim, num_valid_actions, hidden_sizes).to(device)
        self.target_q_net = dueling_net_cls(state_dim, num_valid_actions, hidden_sizes).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.num_actions = num_valid_actions
        self.total_steps = 0

    def state_dict(self):
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state):
        self.q_net.load_state_dict(state["q_net"])
        self.target_q_net.load_state_dict(state["target_q_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = state.get("total_steps", 0)

    @torch.no_grad()
    def select_action(self, state, action_space):
        """
        state: np.array or torch.Tensor of shape [state_dim]
        action_space: ReducedBinaryActionSpace (to convert index->bits)
        Returns: (index, bits)
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_net(s)                          # [1, num_actions]
        idx = int(q.argmax(dim=1).item())

        bits = action_space.index_to_action(idx)       # [n_bits]
        return idx, bits
    
    def reset_noise(self):
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

    def update(self, buffer, batch_size):
        # Sample from PER buffer
        states, actions, rewards, next_states, dones, weights, idxs = buffer.sample(batch_size)
        self.total_steps += 1

        # Compute Q(s,a)
        q = self.q_net(states)                          # [B, A]
        q_sa = q.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            # Double DQN: online net selects action
            q_next_online = self.q_net(next_states)
            a_star = q_next_online.argmax(dim=1)

            # Target net evaluates action
            q_next_target = self.target_q_net(next_states)
            q_s2_a_star = q_next_target.gather(1, a_star.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * (1.0 - dones) * q_s2_a_star

        # TD error
        td_errors = q_sa - target

        # Weighted loss
        loss = (weights * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities in buffer
        buffer.update_priorities(idxs, td_errors)

        return loss.item()


    def update_target(self):
        if self.tau >= 1.0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            # soft update
            with torch.no_grad():
                for tp, p in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)





