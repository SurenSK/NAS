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
import matplotlib.pyplot as plt

# Deterministic setup
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Search Space Config ---
KERNELS = [3, 5]
CHANNELS = [16, 32, 64]
MAX_LAYERS = 10
G = 16
VOCAB = len(KERNELS) * len(CHANNELS)
EOS = VOCAB
SOS = VOCAB + 1

# --- Constraints (Jetson Nano Envelope - loosely enforced for now) ---
MAX_MEM_BYTES = 2 * 1024**3 
MAX_FLOPS = 10 * 10**9

# ==========================================
# Goal: NAS Controller & Child Architecture
# ==========================================
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
    h, w = 32, 32 
    B = 4 # Bytes per float32
    
    for idx in arch:
        if idx == EOS: break
        k, out_c = KERNELS[idx // len(CHANNELS)], CHANNELS[idx % len(CHANNELS)]
        params += (k * k * in_c + 1) * out_c + (4 * out_c)
        flops += 2 * in_c * k * k * h * w * out_c
        input_map = in_c * h * w * B
        output_map = out_c * h * w * B
        if (input_map + output_map) > max_act_bytes:
            max_act_bytes = input_map + output_map
        in_c = out_c
    
    params += (in_c + 1) * 10
    flops += 2 * in_c * 10
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

# ==========================================
# Goal: Training & Evaluation Subroutines
# ==========================================
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
    final_train_loss = 10.0
    
    for i in range(0, data_tr.size(0), 256):
        idx = indices[i:i+256].to(d)
        x, y = data_tr[idx], target_tr[idx]
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        final_train_loss = loss.item()
    
    net.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, data_te.size(0), 1024):
             out = net(data_te[i:i+1024])
             correct += (out.argmax(1) == target_te[i:i+1024]).sum().item()
    
    val_acc = correct / data_te.size(0)
    # Extrapolation Heuristic
    projected_score = val_acc + 0.1 * max(0, (1.0 - final_train_loss))
    return max(0.0, projected_score)

# ==========================================
# Goal: Visualization
# ==========================================
def plot_dashboard(model_history, step_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Step vs Avg Accuracy (with P-val highlights) ---
    steps = [s['step'] for s in step_history]
    avg_accs = [s['avg_acc'] for s in step_history]
    p_vals = [s['p_val'] for s in step_history]
    
    ax = axes[0]
    ax.plot(steps, avg_accs, color='black', linewidth=2, label='Avg Acc')
    
    # Add background highlights for significant steps
    for i, p in enumerate(p_vals):
        if p <= 0.05:
            # Draw rectangle around this step interval
            ax.axvspan(steps[i]-0.5, steps[i]+0.5, facecolor='lightgreen', alpha=0.4)
            
    ax.set_xlabel("Step #")
    ax.set_ylabel("Average Reward (Proj. Accuracy)")
    ax.set_title("Search Stability & Significance (Green: p <= 0.05)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    # --- Prep Data for Scatter Plots (Filter only evaluated models) ---
    valid_models = [h for h in model_history if h['acc'] > 0]
    scatter_steps = [h['step'] for h in valid_models]
    scatter_accs = [h['acc'] for h in valid_models]
    # Convert units for readability
    scatter_flops_M = [h['flops'] / 1e6 for h in valid_models]
    scatter_mem_MB = [h['mem'] / 1024**2 for h in valid_models]
    
    cmap_monochrome = 'Blues'

    # --- Plot 2: FLOPs vs Accuracy Scatter ---
    ax = axes[1]
    sc1 = ax.scatter(scatter_flops_M, scatter_accs, c=scatter_steps, cmap=cmap_monochrome, 
                     alpha=0.7, edgecolors='k', linewidth=0.5, s=40)
    ax.set_xlabel("Compute (MFLOPs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto: Compute vs Accuracy")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=ax, label='Step #')

    # --- Plot 3: Memory vs Accuracy Scatter ---
    ax = axes[2]
    sc2 = ax.scatter(scatter_mem_MB, scatter_accs, c=scatter_steps, cmap=cmap_monochrome, 
                     alpha=0.7, edgecolors='k', linewidth=0.5, s=40)
    ax.set_xlabel("Memory Footprint (MB)")
    # Y-label redundant here
    ax.set_title("Pareto: Space vs Accuracy")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc2, ax=ax, label='Step #')
    
    plt.tight_layout()
    plt.savefig('nas_dashboard.png')
    print("\nSaved visualization to nas_dashboard.png")

# ==========================================
# Goal: Main Execution Loop
# ==========================================
import json

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
    
    # Data containers
    base_acc = []
    model_history = [] 
    step_history = []
    
    # Global tracker for the absolute best model across all steps
    global_best_acc = 0.0
    global_best_arch = []
    global_best_state = None

    SEARCH_STEPS = 50
    print(f"Starting Final Search for {SEARCH_STEPS} steps...")
    
    for step in range(SEARCH_STEPS):
        t0 = time.time()
        actions = ctrl.sample(G)
        
        valid_indices = []
        valid_nets = []
        valid_seeds = []
        final_rewards = [0.0] * G
        
        this_step_models = []
        
        for i, a in enumerate(actions):
            p_count, flops, mem_bytes = get_stats(a.tolist())
            
            penalty = 0.0
            if mem_bytes > MAX_MEM_BYTES: penalty += (mem_bytes - MAX_MEM_BYTES) / MAX_MEM_BYTES
            if flops > MAX_FLOPS:         penalty += (flops - MAX_FLOPS) / MAX_FLOPS
            
            model_record = {
                'step': step,
                'flops': flops,
                'mem': mem_bytes,
                'acc': 0.0 
            }
            this_step_models.append(model_record)

            if penalty > 0:
                final_rewards[i] = -0.1 * penalty 
            else:
                valid_indices.append(i)
                valid_nets.append(Child(a.tolist()).to(d))
                valid_seeds.append(step * 1000 + i)

        if valid_nets:
            futs = [pool.submit(train_and_score, net, s) for net, s in zip(valid_nets, valid_seeds)]
            results = [f.result() for f in futs]
            for idx, acc in zip(valid_indices, results):
                final_rewards[idx] = acc
                this_step_models[idx]['acc'] = acc
                
                # Capture Global Best
                if acc > global_best_acc:
                    global_best_acc = acc
                    global_best_arch = actions[idx].tolist()
                    # We need to grab the state dict from the specific net that won
                    # Note: 'valid_nets' list index corresponds to 'valid_indices' order
                    net_idx = valid_indices.index(idx)
                    global_best_state = valid_nets[net_idx].state_dict()
        
        model_history.extend(this_step_models)
        
        rewards_tensor = torch.tensor(final_rewards, device=d)
        avg_reward = rewards_tensor.mean().item()
        
        valid_rewards = [r for r in final_rewards if r > 0]
        if step == 0:
            base_acc = valid_rewards if valid_rewards else [0.1]
            p_val = 1.0
        elif len(valid_rewards) > 1:
            _, p_val = ttest_ind(base_acc, valid_rewards, equal_var=False, alternative='less')
        else:
            p_val = 1.0
            
        step_history.append({'step': step, 'avg_acc': avg_reward, 'p_val': p_val})

        if rewards_tensor.std() == 0: adv = torch.zeros_like(rewards_tensor)
        else: adv = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        loss = -(ctrl.log_probs(actions) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        best_idx = rewards_tensor.argmax()
        p, f, m = get_stats(actions[best_idx].tolist())
        
        print(f"t+{time.time()-t0:.2f}s | Step {step:02d} | Avg: {avg_reward:.4f} | P-Val: {p_val:.3f} | Best: {rewards_tensor[best_idx]:.4f} (M:{m/1024**2:.1f}MB F:{f/1e6:.0f}M)")

    # --- Save Artifacts ---
    print(f"\nSearch Complete. Global Best Accuracy: {global_best_acc:.4f}")
    print(f"Best Architecture: {global_best_arch}")
    
    # Save Architecture
    with open('best_arch.json', 'w') as f:
        json.dump(global_best_arch, f)
    
    # Save Weights
    if global_best_state:
        torch.save(global_best_state, 'best_model.pth')
        print("Saved best model to 'best_model.pth' and 'best_arch.json'")

    # Viz
    plot_dashboard(model_history, step_history)