"""CPU smoke test for ablation mechanisms. Not part of the main codebase."""
import os, sys, time, math
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

# ---- Stub out FA3 with PyTorch SDPA ----
class _SDPAStub:
    @staticmethod
    def flash_attn_func(q, k, v, causal=True, window_size=None):
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return y.transpose(1,2)

class _FA3Module:
    flash_attn_interface = _SDPAStub()

# Inject stub before importing train.py model code
import types
fake_kernels = types.ModuleType("kernels")
fake_kernels.get_kernel = lambda *a, **kw: _FA3Module()
sys.modules["kernels"] = fake_kernels

# Monkey-patch torch.cuda so the module-level code doesn't crash
_orig_cuda = torch.cuda
class _FakeCuda:
    def get_device_capability(self): return (0, 0)
    def synchronize(self): pass
    def max_memory_allocated(self): return 0
    def manual_seed(self, s): pass
    def __getattr__(self, name): return lambda *a, **kw: None
torch.cuda = _FakeCuda()

# Now read prepare.py constants (we need Tokenizer, etc.)
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer

# Restore real cuda module (we won't use it, but in case)
torch.cuda = _orig_cuda

# ---- Import model classes from train.py source ----
# We'll exec just the model/config portion, stopping before the runtime code

with open("train.py") as f:
    src = f.read()

# Extract everything from the model section up to (but not including) the
# module-level runtime code that starts with "t_start = time.time()"
lines = src.split("\n")
model_end = next(i for i, l in enumerate(lines) if l.startswith("t_start = time.time()"))
model_src = "\n".join(lines[:model_end])

# Replace the problematic imports at the top
model_src = model_src.replace("from kernels import get_kernel", "# stubbed out")
model_src = model_src.replace("cap = torch.cuda.get_device_capability()", "cap = (0,0)")
model_src = model_src.replace('repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"', "")
model_src = model_src.replace("fa3 = get_kernel(repo).flash_attn_interface", "fa3 = _SDPAStub()")
model_src = model_src.replace("from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb", "# stubbed out")

# Patch .view() calls that break on non-contiguous CPU tensors
model_src = model_src.replace("logits.view(-1,", "logits.reshape(-1,")
model_src = model_src.replace("targets.view(-1)", "targets.reshape(-1)")

# Execute model code into this namespace
exec(model_src, globals())

# ---- CPU dataloader (no pinned memory, no cuda) ----
import pyarrow.parquet as pq

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

def cpu_dataloader(tokenizer, B, T, split):
    """Minimal CPU dataloader. Yields (x, y, epoch)."""
    parquet_paths = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR) if f.endswith(".parquet")
    )
    val_path = os.path.join(DATA_DIR, "shard_06542.parquet")
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]

    bos = tokenizer.get_bos_token_id()
    epoch = 1
    while True:
        for fp in parquet_paths:
            pf = pq.ParquetFile(fp)
            for rg_idx in range(pf.num_row_groups):
                texts = pf.read_row_group(rg_idx).column("text").to_pylist()
                tokens = tokenizer.encode(texts, prepend=bos)
                # flatten into one long stream
                flat = []
                for t in tokens:
                    flat.extend(t)
                flat = torch.tensor(flat, dtype=torch.long)
                # yield batches
                total = (len(flat) // (B * (T + 1))) * B * (T + 1)
                for start in range(0, total, B * (T + 1)):
                    chunk = flat[start:start + B * (T + 1)].view(B, T + 1)
                    yield chunk[:, :T], chunk[:, 1:T+1], epoch
        epoch += 1

# ---- Run ----
DEPTH = int(os.environ.get("DEPTH", 3))
SEQ = int(os.environ.get("SEQ", 128))
BATCH = int(os.environ.get("BATCH", 4))
VOCAB = 8192  # match trained tokenizer
ASPECT_RATIO = 64
HEAD_DIM = int(os.environ.get("HEAD_DIM", 64))
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", 30))

def make_cfg(**kw):
    d = DEPTH * ASPECT_RATIO
    d = ((d + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    nh = d // HEAD_DIM
    return GPTConfig(
        sequence_len=SEQ, vocab_size=VOCAB, n_layer=DEPTH,
        n_head=nh, n_kv_head=nh, n_embd=d, window_pattern="L", **kw
    )

def run_config(name, **kw):
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    cfg = make_cfg(**kw)
    print(f"  depth={DEPTH} dim={cfg.n_embd} heads={cfg.n_head} head_dim={HEAD_DIM} seq={SEQ}")

    model = GPT(cfg)
    # init_weights casts embeddings to bf16, which causes stride issues on CPU
    # after .float(). Use contiguous float conversion instead.
    model.init_weights()
    for p in model.parameters():
        p.data = p.data.float().contiguous()
    for name_, buf in model.named_buffers():
        buf.data = buf.data.float().contiguous()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"  params: {nparams:,}")

    # Simple Adam for CPU test (Muon requires stacked same-shape params)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    tokenizer = Tokenizer.from_directory()
    loader = cpu_dataloader(tokenizer, BATCH, SEQ, "train")

    t0 = time.time()
    step = 0
    while True:
        x, y, epoch = next(loader)
        loss = model(x, y)
        if torch.isnan(loss):
            print(f"  step {step}: NaN loss -- FAIL")
            return False, {}
        loss.backward()
        opt.step()
        opt.zero_grad()
        dt = time.time() - t0
        if step % 5 == 0:
            print(f"\r  step {step:3d} | loss {loss.item():.4f} | {dt:.1f}s", end="", flush=True)
        step += 1
        if dt > TIME_LIMIT:
            break

    final_loss = loss.item()
    total_tokens = step * BATCH * SEQ
    tok_sec = int(total_tokens / dt) if dt > 0 else 0
    print(f"\n  final: {step} steps in {dt:.1f}s, loss={final_loss:.4f}, tok/sec={tok_sec:,}")

    # Quick val check
    model.eval()
    with torch.no_grad():
        vl = cpu_dataloader(tokenizer, BATCH, SEQ, "val")
        vx, vy, _ = next(vl)
        val_loss = model(vx, vy).item()
    print(f"  val_loss: {val_loss:.4f}")
    return True, dict(train_loss=final_loss, val_loss=val_loss, steps=step, params=nparams, tok_sec=tok_sec)

# ---- Test all ablation configs from the plan ----
configs = [
    ("baseline",                     dict()),
    ("5c + 1 orig RRPRAM",           dict(n_rrpram_original=1)),
    ("5c + 1 RoPE-RRPRAM (shared)",  dict(n_rrpram=1, rrpram_shared_v=True)),
    ("5c + 1 RoPE-RRPRAM (sep v)",   dict(n_rrpram=1, rrpram_shared_v=False)),
    ("5c + 1 Janus",                 dict(n_janus=1)),
]

results = {}
for name, kw in configs:
    ok, metrics = run_config(name, **kw)
    results[name] = (ok, metrics)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'Config':40s}  {'Pass':4s}  {'Train':>8s}  {'Val':>8s}  {'Steps':>5s}  {'tok/sec':>8s}  {'Params':>10s}")
for name, (ok, m) in results.items():
    if m:
        print(f"  {name:38s}  {'PASS' if ok else 'FAIL'}  {m['train_loss']:8.4f}  {m['val_loss']:8.4f}  {m['steps']:5d}  {m['tok_sec']:>8,}  {m['params']:10,}")
    else:
        print(f"  {name:38s}  FAIL")
