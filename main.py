import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
import time

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KERNELS = [3, 5]
CHANNELS = [16, 32, 64]
LAYERS = 3 
G = 16
VOCAB = len(KERNELS) * len(CHANNELS)
SOS = VOCAB
BATCH_SIZE = 256

class ARController(nn.Module):
    def __init__(self, dm=64, nh=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 1, dm)
        self.pos = nn.Parameter(torch.randn(1, LAYERS + 1, dm))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, 256, batch_first=True), 2)
        self.head = nn.Linear(dm, VOCAB)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=d)
        return self.head(self.tf(self.emb(x) + self.pos[:, :x.size(1)], mask=mask))

    def sample(self, n):
        curr = torch.full((n, 1), SOS, dtype=torch.long, device=d)
        for _ in range(LAYERS):
            logits = self.forward(curr)[:, -1, :]
            curr = torch.cat([curr, Categorical(logits=logits).sample().unsqueeze(1)], 1)
        return curr[:, 1:]

    def log_probs(self, actions):
        inp = torch.cat([torch.full((actions.size(0), 1), SOS, device=d), actions], 1)[:, :-1]
        return Categorical(logits=self.forward(inp)).log_prob(actions).sum(1)

class Child(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers, in_c = [], 3
        for idx in arch:
            k, out = KERNELS[idx // len(CHANNELS)], CHANNELS[idx % len(CHANNELS)]
            layers += [nn.Conv2d(in_c, out, k, padding=k//2), nn.ReLU(), nn.BatchNorm2d(out)]
            in_c = out
        self.net = nn.Sequential(*layers)
        self.gap, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(in_c, 10)

    def forward(self, x):
        return self.fc(self.gap(self.net(x)).flatten(1))

# Global data containers
data_tr, target_tr = None, None
data_te, target_te = None, None

def train_and_score(arch):
    # Train
    net = Child(arch).to(d)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    
    # Manual shuffling and batching on GPU tensors
    perm = torch.randperm(data_tr.size(0), device=d)
    for i in range(0, data_tr.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x, y = data_tr[idx], target_tr[idx]
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    
    # Eval
    net.eval()
    with torch.no_grad():
        # Batching test data to save memory
        correct = 0
        for i in range(0, data_te.size(0), 1024):
             out = net(data_te[i:i+1024])
             correct += (out.argmax(1) == target_te[i:i+1024]).sum().item()
    return correct / data_te.size(0)

if __name__ == "__main__":
    # Pre-load Data to GPU (VRAM) to bypass DataLoader overhead in threads
    print("Loading data to VRAM...")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds_tr = datasets.CIFAR10('./data', True, transform=tf, download=True)
    ds_te = datasets.CIFAR10('./data', False, transform=tf, download=True)
    
    # Convert entire datasets to CUDA tensors once
    data_tr = torch.stack([s[0] for s in ds_tr]).to(d)
    target_tr = torch.tensor([s[1] for s in ds_tr], device=d)
    data_te = torch.stack([s[0] for s in ds_te]).to(d)
    target_te = torch.tensor([s[1] for s in ds_te], device=d)

    ctrl = ARController().to(d)
    opt = optim.Adam(ctrl.parameters(), lr=5e-4)
    
    # Thread pool for parallel execution
    pool = ThreadPoolExecutor(max_workers=4) 

    print("Starting Search...")
    for step in range(50):
        t0 = time.time()
        actions = ctrl.sample(G)
        
        # Parallel Execution
        # We pass the architecture list to threads. 
        # CUDA streams handle the concurrency implicitly.
        futs = [pool.submit(train_and_score, a.tolist()) for a in actions]
        rewards = [f.result() for f in futs]
        
        accuracies = torch.tensor(rewards, device=d)
        
        if accuracies.std() == 0: adv = torch.zeros_like(accuracies)
        else: adv = (accuracies - accuracies.mean()) / (accuracies.std() + 1e-8)
        
        loss = -(ctrl.log_probs(actions) * adv).mean()
        
        opt.zero_grad(); loss.backward(); opt.step()

        best_idx = accuracies.argmax()
        print(f"t+{time.time()-t0:.2f}s Step {step:02d} | Avg: {accuracies.mean():.4f} | Best: {accuracies[best_idx]:.4f} | Arch: {actions[best_idx].tolist()}")