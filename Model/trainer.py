import os
import torch
import json
from RLModel import DQNAgent, ReducedBinaryActionSpace, DuelingQNetwork, PrioritizedReplayBuffer


def save_checkpoint(model, folder="checkpoints", prefix="w"):
    os.makedirs(folder, exist_ok=True)

    # Find existing files
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".pth")]
    if existing:
        # Extract numeric suffixes
        indices = [int(f[len(prefix)+1:-4]) for f in existing if f[len(prefix)+1:-4].isdigit()]
        next_idx = max(indices) + 1 if indices else 0
    else:
        next_idx = 0

    filename = f"{prefix}_{next_idx}.pth"
    path = os.path.join(folder, filename)

    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")
    return path

def consume_episodes_once(folder="episodes", seen=None):
    """
    Scan the folder once for new .json episodes.
    Returns a list of loaded episodes.
    
    Args:
        folder (str): directory to scan
        seen (set): set of filenames already processed (mutable, maintained by caller)
    """
    if seen is None:
        seen = set()

    new_episodes = []
    if not os.path.exists(folder):
        return new_episodes, seen

    for fname in os.listdir(folder):
        if fname.endswith(".json") and fname not in seen:
            path = os.path.join(folder, fname)
            with open(path) as f:
                episode = json.load(f)
            new_episodes.append(episode)
            seen.add(fname)

    return new_episodes, seen

def train_from_buffer(
    agent,
    buffer,
    batch_size=64,
    train_steps=10,
    target_update_interval=1000,
):
    """
    Run a block of training updates using samples from the replay buffer.

    Args:
        agent: DQNAgent instance
        buffer: replay buffer (already filled by actor process)
        batch_size: minibatch size
        train_steps: how many gradient steps to run this call
        target_update_interval: how often to sync target network
    """
    losses = []
    for step in range(train_steps):
        if len(buffer) < batch_size:
            break  # not enough data yet

        loss = agent.update(buffer, batch_size)
        losses.append(loss)

        # Periodically update target network
        if agent.total_steps % target_update_interval == 0:
            agent.update_target()

    return losses

def no_opposites(actions):
    left, right, up, down = actions[:,0], actions[:,1], actions[:,2], actions[:,3]
    return (left + right <= 1) & (up + down <= 1)

space = ReducedBinaryActionSpace(n_bits=4, constraints=no_opposites)

# Agent and buffer
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 128                    # example
num_valid_actions = len(space)
agent = DQNAgent(state_dim, num_valid_actions, DuelingQNetwork, device=device)
buffer = PrioritizedReplayBuffer(capacity=100_000, state_dim=state_dim, device=device)

num_epochs = 10

for epoch in range(num_epochs):
    losses = train_from_buffer(agent, buffer, batch_size=64, train_steps=50)

    if losses:
        print(f"Epoch {epoch}: mean loss {sum(losses)/len(losses):.4f}")
