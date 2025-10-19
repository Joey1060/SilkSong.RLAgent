import time
import torch
import os
import json
import uuid
import sys
import re
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
        msg = self.client.receive()
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

class EpisodeDumper:
    def __init__(self, folder="episodes"):
        self.folder = folder
        self.counter = self.init_episode_counter(folder)
    
    def init_episode_counter(self, folder="episodes"):
        os.makedirs(folder, exist_ok=True)
        existing = [
            int(re.match(r"(\d+)\.json", f).group(1))
            for f in os.listdir(folder)
            if re.match(r"^\d+\.json$", f)
        ]
        return max(existing, default=0)

    def dump(self, episode_data):
        self.counter += 1
        final_name = os.path.join(self.folder, f"{self.counter}.json")
        tmp_name = final_name + ".tmp"

        with open(tmp_name, "w") as f:
            json.dump(episode_data, f)

        os.replace(tmp_name, final_name)
        print(f"Episode saved: {final_name}")

def no_opposites(actions):
    left, right, up, down = actions[:,0], actions[:,1], actions[:,2], actions[:,3]
    return (left + right <= 1) & (up + down <= 1)


training_mode = True
if (len(sys.argv) == 2):
    if (sys.argv[1] == '--eval'):
        training_mode = False

space = ReducedBinaryActionSpace(n_bits=4, constraints=no_opposites)

# Agent and buffer
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 5                    # example
num_valid_actions = len(space)
agent = DQNAgent(state_dim, num_valid_actions, DuelingQNetwork, device=device)
env = RLEnv()
cur_checkpoint = None

dumper = EpisodeDumper()

if training_mode:
    while (True):
        latest_checkpoint = agent.find_latest_checkpoint()
        # if (latest_checkpoint is None):
        #     time.sleep(5)
        #     continue
        if (latest_checkpoint != cur_checkpoint):
            print(f"new checkpoint found: {latest_checkpoint}, loading....")
            agent.load_checkpoint(latest_checkpoint)
            cur_checkpoint = latest_checkpoint

        episode_data = []
        next_state = env.reset()
        done = False
        while (not done):
            # print("?")
            agent.reset_noise()
            _, action = agent.select_action(next_state, space)
            action_list = action.numpy()
            action_mask = 0
            for i in range(len(action_list)):
                action_mask = action_mask | ((1 & action_list[i]) << i)
            # print(action_list, action_mask)
            prev_state, prev_action, reward, next_state, done = env.step(action_mask)
            episode_data.append([prev_state, space.action_to_index(action), reward, next_state, done])
        print(f"episode finished, dumping data....")
        dumper.dump(episode_data)
else:
    with torch.no_grad():
        latest_checkpoint = agent.find_latest_checkpoint()
        if (latest_checkpoint is not None):
            agent.load_checkpoint(latest_checkpoint)
        agent.q_net.eval()
        next_state = env.reset()
        done = False
        while (not done):
            # print("?")
            agent.reset_noise()
            _, action = agent.select_action(next_state, space)
            print(action)
            action_list = action.numpy()
            action_mask = 0
            for i in range(len(action_list)):
                action_mask = action_mask | ((1 & action_list[i]) << i)
            # print(action_list, action_mask)
            prev_state, prev_action, reward, next_state, done = env.step(action_mask)
        print(f"episode finished")
env.close()