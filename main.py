import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KERNELS = [3, 5]
CHANNELS = [16, 32, 64]
MAX_LAYERS = 10
G = 16
VOCAB = len(KERNELS) * len(CHANNELS)
EOS = VOCAB
SOS = VOCAB + 1

class ARController(nn.Module):
    def __init__(self, dm=64, nh=4):
        super().__init__()
        # Input Embedding: Must handle Actions (0-5), EOS (6), AND SOS (7)
        self.emb = nn.Embedding(SOS + 1, dm)
        self.pos = nn.Parameter(torch.randn(1, MAX_LAYERS + 1, dm))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, 256, batch_first=True), 2)
        
        # Output Head: Can predict Actions (0-5) or EOS (6). 
        # Must strictly EXCLUDE SOS (7) from being a valid output.
        self.head = nn.Linear(dm, EOS + 1)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=d)
        return self.head(self.tf(self.emb(x) + self.pos[:, :x.size(1)], mask=mask))

    def sample(self, n):
        curr = torch.full((n, 1), SOS, dtype=torch.long, device=d)
        for _ in range(MAX_LAYERS):
            # Forward pass now returns logits for [0..6] (Actions + EOS)
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

def train_and_score(arch):
    net = Child(arch).to(d)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    perm = torch.randperm(data_tr.size(0), device=d)
    for i in range(0, data_tr.size(0), 256):
        idx = perm[i:i+256]
        x, y = data_tr[idx], target_tr[idx]
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    
    net.eval()
    with torch.no_grad():
        correct = 0
        for i in range(0, data_te.size(0), 1024):
             out = net(data_te[i:i+1024])
             correct += (out.argmax(1) == target_te[i:i+1024]).sum().item()
    return correct / data_te.size(0)

if __name__ == "__main__":
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

    print("Starting Search...")
    for step in range(50):
        actions = ctrl.sample(G)
        futs = [pool.submit(train_and_score, a.tolist()) for a in actions]
        rewards = [f.result() for f in futs]
        
        accuracies = torch.tensor(rewards, device=d)
        
        if step == 0:
            base_acc = rewards
            p_val = 1.0
        else:
            _, p_val = ttest_ind(base_acc, rewards, equal_var=False)

        if accuracies.std() == 0: adv = torch.zeros_like(accuracies)
        else: adv = (accuracies - accuracies.mean()) / (accuracies.std() + 1e-8)
        
        loss = -(ctrl.log_probs(actions) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        best_idx = accuracies.argmax()
        print(f"Step {step:02d} | Avg: {accuracies.mean():.4f} | P-Val: {p_val:.3e} | Arch: {actions[best_idx].tolist()}")