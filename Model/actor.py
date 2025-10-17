import torch
import os
import json
import uuid
from RLModel import DQNAgent, ReducedBinaryActionSpace, DuelingQNetwork

def find_latest_checkpoint(folder="weights", prefix="w"):
    """
    Scan the folder for files like w_0.pth, w_1.pth, ...
    Return the path to the highest-index file, or None if none exist.
    """
    if not os.path.exists(folder):
        return None

    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".pth")]
    if not existing:
        return None

    # Extract numeric suffixes
    indices = [(int(f[len(prefix)+1:-4]), f) for f in existing if f[len(prefix)+1:-4].isdigit()]
    if not indices:
        return None

    latest_idx, latest_file = max(indices, key=lambda x: x[0])
    return os.path.join(folder, latest_file)

def load_checkpoint(model, path, device="cpu"):
    """
    Load weights from a given checkpoint path into the model.
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {path}")
    return model

def dump_episode(episode_data, folder="episodes"):
    os.makedirs(folder, exist_ok=True)
    tmp_name = os.path.join(folder, f"{uuid.uuid4()}.tmp")
    final_name = tmp_name.replace(".tmp", ".json")

    # Write to temp file
    with open(tmp_name, "w") as f:
        json.dump(episode_data, f)

    # Atomic rename → trainer only sees .json when fully written
    os.rename(tmp_name, final_name)
    print(f"Episode saved: {final_name}")

# with torch.no_grad():
    

def no_opposites(actions):
    left, right, up, down = actions[:,0], actions[:,1], actions[:,2], actions[:,3]
    return (left + right <= 1) & (up + down <= 1)

space = ReducedBinaryActionSpace(n_bits=4, constraints=no_opposites)

# Agent and buffer
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 128                    # example
num_valid_actions = len(space)
agent = DQNAgent(state_dim, num_valid_actions, DuelingQNetwork, device=device)
