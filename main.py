import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind
import time
import random
import numpy as np
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KERNELS = [3, 5]
CHANNELS = [16, 32, 64]
MAX_LAYERS = 10
G = 16
VOCAB = len(KERNELS) * len(CHANNELS)
EOS = VOCAB
SOS = VOCAB + 1

# --- Constraints (Jetson Nano Envelope) ---
# 2 GB RAM (Bytes) and 10 GFLOPs (FP32 operations)
MAX_MEM_BYTES = 2 * 1024**3 
MAX_FLOPS = 10 * 10**9

class ARController(nn.Module):
    def __init__(self, dm=64, nh=4):
        super().__init__()
        self.emb = nn.Embedding(SOS + 1, dm)
        self.pos = nn.Parameter(torch.randn(1, MAX_LAYERS + 1, dm))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, 256, batch_first=True), 2)
        self.head = nn.Linear(dm, EOS + 1)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=d)
        return self.head(self.tf(self.emb(x) + self.pos[:, :x.size(1)], mask=mask))

    def sample(self, n):
        curr = torch.full((n, 1), SOS, dtype=torch.long, device=d)
        for _ in range(MAX_LAYERS):
            logits = self.forward(curr)[:, -1, :]
            curr = torch.cat([curr, Categorical(logits=logits).sample().unsqueeze(1)], 1)
        return curr[:, 1:]

    def log_probs(self, actions):
        inp = torch.cat([torch.full((actions.size(0), 1), SOS, device=d), actions], 1)[:, :-1]
        return Categorical(logits=self.forward(inp)).log_prob(actions).sum(1)

def get_stats(arch):
    params = 0
    flops = 0
    max_act_bytes = 0
    
    in_c = 3
    h, w = 32, 32 # CIFAR Resolution
    
    # Bytes per float32
    B = 4 
    
    for idx in arch:
        if idx == EOS: break
        k, out_c = KERNELS[idx // len(CHANNELS)], CHANNELS[idx % len(CHANNELS)]
        
        # 1. Parameter Count
        params += (k * k * in_c + 1) * out_c + (4 * out_c) # Conv + BN
        
        # 2. FLOPs
        flops += 2 * in_c * k * k * h * w * out_c
        
        # 3. Peak Memory (Activation + Input for this layer)
        # In inference, we need to hold Input (in_c) and write to Output (out_c)
        input_map = in_c * h * w * B
        output_map = out_c * h * w * B
        
        # Track the peak memory usage of the widest layer interaction
        if (input_map + output_map) > max_act_bytes:
            max_act_bytes = input_map + output_map
            
        in_c = out_c
    
    # Head
    params += (in_c + 1) * 10
    flops += 2 * in_c * 10
    
    # Total RAM = Model Weights + Peak Activations
    total_mem = (params * B) + max_act_bytes
    
    return params, flops, total_mem

class Child(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers, in_c = [], 3
        for idx in arch:
            if idx == EOS: break
            k, out = KERNELS[idx // len(CHANNELS)], CHANNELS[idx % len(CHANNELS)]
            layers += [nn.Conv2d(in_c, out, k, padding=k//2), nn.ReLU(), nn.BatchNorm2d(out)]
            in_c = out
        self.net = nn.Sequential(*layers)
        self.gap, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(in_c, 10)

    def forward(self, x):
        return self.fc(self.gap(self.net(x)).flatten(1))

data_tr, target_tr = None, None
data_te, target_te = None, None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def train_and_score(net, seed):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    
    indices = torch.randperm(data_tr.size(0), generator=gen)
    final_train_loss = 0.0

    for i in range(0, data_tr.size(0), 256):
        idx = indices[i:i+256].to(d)
        x, y = data_tr[idx], target_tr[idx]
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        final_train_loss = loss.item() # Capture loss of the final batch
    
    net.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, data_te.size(0), 1024):
             out = net(data_te[i:i+1024])
             correct += (out.argmax(1) == target_te[i:i+1024]).sum().item()
    
    val_acc = correct / data_te.size(0)

    # Simple Extrapolation Heuristic:
    # We reward models that have lower training loss (better fit) by adding a small projection bonus.
    # If final_train_loss is 0.0 (perfect fit), we add 0.1 to the score.
    # If final_train_loss is > 1.0 (underfitting), this acts as a penalty.
    projected_score = val_acc + 0.1 * (1.0 - final_train_loss)

    return max(0.0, projected_score)

if __name__ == "__main__":
    set_seed(42)

    print("Loading data to VRAM...")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds_tr = datasets.CIFAR10('./data', True, transform=tf, download=True)
    ds_te = datasets.CIFAR10('./data', False, transform=tf, download=True)
    
    data_tr = torch.stack([s[0] for s in ds_tr]).to(d)
    target_tr = torch.tensor([s[1] for s in ds_tr], device=d)
    data_te = torch.stack([s[0] for s in ds_te]).to(d)
    target_te = torch.tensor([s[1] for s in ds_te], device=d)

    ctrl = ARController().to(d)
    opt = optim.Adam(ctrl.parameters(), lr=5e-4)
    pool = ThreadPoolExecutor(max_workers=4)
    base_acc = []

    print(f"Starting Search (Max RAM: {MAX_MEM_BYTES/1024**3:.2f}GB, Max FLOPs: {MAX_FLOPS/1e9:.2f}G)...")
    
    for step in range(50):
        t0 = time.time()
        actions = ctrl.sample(G)
        
        # 1. Filter & Calculate Hinge Loss
        valid_indices = []
        valid_nets = []
        valid_seeds = []
        final_rewards = [0.0] * G
        
        for i, a in enumerate(actions):
            # Calculate stats analytically
            p_count, flops, mem_bytes = get_stats(a.tolist())
            
            penalty = 0.0
            if mem_bytes > MAX_MEM_BYTES: penalty += (mem_bytes - MAX_MEM_BYTES) / MAX_MEM_BYTES
            if flops > MAX_FLOPS:         penalty += (flops - MAX_FLOPS) / MAX_FLOPS
            
            if penalty > 0:
                final_rewards[i] = -0.1 * penalty # Weighted penalty
            else:
                valid_indices.append(i)
                valid_nets.append(Child(a.tolist()).to(d))
                valid_seeds.append(step * 1000 + i)

        # 2. Train ONLY valid models
        if valid_nets:
            futs = [pool.submit(train_and_score, net, s) for net, s in zip(valid_nets, valid_seeds)]
            results = [f.result() for f in futs]
            for idx, acc in zip(valid_indices, results):
                final_rewards[idx] = acc
        
        # 3. RL Update
        rewards_tensor = torch.tensor(final_rewards, device=d)
        
        valid_rewards = [r for r in final_rewards if r > 0]
        if step == 0:
            base_acc = valid_rewards if valid_rewards else [0.1]
            p_val = 1.0
        elif len(valid_rewards) > 1:
            _, p_val = ttest_ind(base_acc, valid_rewards, equal_var=False)
        else:
            p_val = 1.0

        if rewards_tensor.std() == 0: adv = torch.zeros_like(rewards_tensor)
        else: adv = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        loss = -(ctrl.log_probs(actions) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        best_idx = rewards_tensor.argmax()
        p, f, m = get_stats(actions[best_idx].tolist())
        
        # print(f"t+{time.time()-t0:.2f}s | Step {step:02d} | Avg: {rewards_tensor.mean():.4f} | Valid: {len(valid_indices)}/{G} | Best: {rewards_tensor[best_idx]:.4f} (M:{m/1024**2:.1f}MB F:{f/1e6:.0f}M)")
        
        print(f"t+{time.time()-t0:.2f}s | Step {step:02d} | Avg: {rewards_tensor.mean():.4f} | P-Val: {p_val:.3f} | Arch: {actions[best_idx].tolist()} {rewards_tensor[best_idx]:.4f} (M:{m/1024**2:.1f}MB F:{f/1e6:.0f}M)")