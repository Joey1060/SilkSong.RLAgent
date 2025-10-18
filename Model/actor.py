import torch
import os
import json
import uuid
import socket
import json
from RLModel import DQNAgent, ReducedBinaryActionSpace, DuelingQNetwork

class RLClient:
    def __init__(self, host="127.0.0.1", port=8001):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.buffer = b""

    def send(self, obj: dict):
        """Send a JSON object to the C# server."""
        data = json.dumps(obj).encode("utf-8")
        # Add newline as a delimiter (important!)
        self.sock.sendall(data + b"\n")

    def receive(self) -> dict:
        """Receive a JSON object from the C# server."""
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self.buffer += chunk

        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self):
        self.sock.close()

class RLEnv:
    def __init__(self, host="127.0.0.1", port=8001):
        # create a socket client instance
        self.client = RLClient(host, port)

    def reset(self):
        """Start a new episode and return the initial state."""
        # send startOB command
        self.client.send({"code": 1})
        print('msg sent')
        msg = self.client.receive()
        print('received', msg)
        # Expect: {"code":3, "transition":{...}}
        state = msg["transition"]["CurState"]
        return state

    def step(self, action: int):
        """Send an action and return (prev_state, prev_action, reward, next_state, done)."""
        self.client.send({"code": 4, "action": int(action)})
        msg = self.client.receive()
        t = msg["transition"]
        next_state = t["CurState"]
        reward = t["Reward"]
        done = bool(t["Done"])
        prev_state = t["PrevState"]
        prev_action = t["Action"]
        return prev_state, prev_action, reward, next_state, done

    def close(self):
        self.client.close()

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
state_dim = 5                    # example
num_valid_actions = len(space)
agent = DQNAgent(state_dim, num_valid_actions, DuelingQNetwork, device=device)
env = RLEnv()

for i in range(2):
    episode_data = []
    next_state = env.reset()
    print(next_state)
    done = False
    while (not done):
        print("?")
        _, action = agent.select_action(next_state, space)
        action_list = action.numpy()
        action_mask = 0
        for i in range(len(action_list)):
            action_mask = action_mask | ((1 & action_list[i]) << i)
        print(action_list, action_mask)
        prev_state, prev_action, reward, next_state, done = env.step(action_mask)
        episode_data.append([prev_state, prev_action, reward, next_state, done])
    dump_episode(episode_data)
env.close()