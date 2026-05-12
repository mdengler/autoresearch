# Ablation Mechanism Log

## 2026-05-12: Attention backend is an explicit measurement coordinate

**Problem**: The non-Hopper fallback fix made the custom SDPA path too invisible. That would mix attention backend effects into mechanism ablation results.

**Findings**: Spark/aarch64 GB10 should use FA2 as the normal non-Hopper backend. The custom SDPA path is useful, but it is a separate backend baseline, not the default mechanism-test path.

**Fix**: `ATTN_BACKEND=auto` now resolves to FA3 on Hopper and FA2 elsewhere. `ATTN_BACKEND=custom_sdpa` remains available and is run once as an extra baseline row by the GPU ablation harness. The final training summary and TSV output now include `attention_backend`, `attention_backend_requested`, and `attention_kernel_repo`.

**Verification**: Local verification covers syntax and the CPU custom-SDPA smoke path only; GPU backend validation still needs Spark.

## 2026-05-11: GPU ablation baseline failed in FA3 on aarch64

**Problem**: `test_ablations_20260509_01_gpu.py` failed before any ablation mechanism ran. The visible tail ended inside `kernels-community/flash-attn3` during the baseline content-attention call, and the wrapper had truncated the actual exception.

**Findings**: The failing path was the non-Hopper `kernels-community/flash-attn3` aarch64 build under `torch.compile`, not Janus or RRPRAM. The baseline was still using `N_JANUS=0`, `N_RRPRAM=0`, and `N_RRPRAM_ORIGINAL=0`.

**Fix**: Initial fix added an explicit backend selector and a PyTorch SDPA fallback. This was superseded on 2026-05-12 so non-Hopper `auto` now resolves to FA2 and the SDPA path is measured separately. `ATTN_BACKEND=fa3` remains available for forced FA3 testing. The GPU ablation harness now prints more failing stderr so future backend failures include the real exception.

**Verification**: GPU verification still needs to run on the target machine. This local session has no NVIDIA driver, so it can only validate syntax and CPU-side behavior.

## 2026-05-09: CPU smoke test (all mechanisms)

**Setup**: depth=3, dim=192, heads=3, head_dim=64, seq=128, batch=4, vocab=8192, Adam lr=1e-3, 30s per config, CPU (no GPU), SDPA instead of FA3.

| Config | Train loss | Val loss | Steps | Params |
|--------|-----------|----------|-------|--------|
| Baseline (3 content) | 7.204 | 7.551 | 212 | 7,618,758 |
| 5c + 1 orig RRPRAM | 6.926 | 7.530 | 218 | 6,570,118 |
| 5c + 1 RoPE-RRPRAM (shared v) | 7.714 | 7.557 | 211 | 6,533,254 |
| 5c + 1 RoPE-RRPRAM (sep v) | 7.218 | 7.511 | 214 | 6,570,118 |
| 5c + 1 Janus | 6.853 | 7.479 | 221 | 6,533,254 |

**Result**: All 5 configs train stably. No NaN, all gradients flow. Loss differences are not meaningful at this scale.

### CPU scale analysis: when would differences be meaningful?

**Short answer**: probably never, on CPU. The CPU test proves the code works and gradients flow. Actual ablation science needs a GPU.

#### The problem is the optimizer, not just scale

The real autoresearch setup uses Muon (orthogonal gradient updates) for matrix params and Adam for embeddings, with per-group LR scheduling. The CPU test uses plain Adam for everything. The mechanisms under test are subtle architectural differences in how attention is computed -- the kind of thing where optimizer choice and learning rate tuning can easily swamp the signal. A mechanism might look dead under Adam but come alive under Muon, or vice versa.

#### What "meaningful" requires

For a loss difference to be meaningful:

1. **Convergence**: enough steps to get past the "everything learns the same unigram statistics" phase. At depth=3/seq=128, all configs hit ~7.2 train loss -- still in the early, undifferentiated part of the learning curve.
2. **Sufficient capacity**: the mechanisms add inductive bias to attention. At depth=3 with 3 heads, there's barely room for content attention, let alone testing whether a different kind of attention helps. The plan calls for depth=12 / 6 heads for a reason.
3. **Sufficient context**: RRPRAM is about position-aware routing patterns. At seq=128, there aren't enough positions for position-specific patterns to matter. Need at least seq=512, ideally 1024+.
4. **Multiple seeds**: the plan calls for 3 seeds for any winner with >0.005 BPB gap. One run proves nothing.

#### CPU step time benchmarks

Measured on this machine (CPU only, SDPA, Adam, no torch.compile):

| Scale | Params | ms/step | Steps/hr |
|-------|--------|---------|----------|
| depth=3, seq=128, batch=4 | 7.6M | 93 | 38,750 |
| depth=6, seq=256, batch=4 | 26M | 593 | 6,072 |
| depth=6, seq=512, batch=2 | 26M | 642 | 5,609 |
| depth=8, seq=512, batch=2 | 50M | 1,350 | 2,667 |
| depth=12, seq=512, batch=1 | 135M | 2,277 | 1,581 |
| depth=12, seq=1024, batch=1 | 135M | 4,236 | 850 |

#### Time estimates for full ablation on CPU

| Scale | Time per config | 5 configs x 3 seeds |
|-------|-----------------|---------------------|
| depth=6, seq=256, ~3k steps | 30 min | 7.5 hours |
| depth=8, seq=512, ~5k steps | 1.9 hours | 28 hours |
| depth=12, seq=512, ~8k steps | 5.1 hours | 76 hours |
| depth=12, seq=1024, ~8k steps | 9.4 hours | 141 hours |

#### Assessment

Depth=8, seq=512 (~2 hours/run, ~30 hours total) is the minimum where any real signal between mechanisms might appear. Even then, results would be less trustworthy than a single 5-minute GPU run with the proper optimizer because:

- **Adam is not Muon.** The optimizer interacts with the architecture.
- **Batch=2 is not 524K tokens.** Small batch noise can obscure small architectural effects.
- **No torch.compile.** CPU can't fuse ops the way the GPU path does.

The CPU test serves its purpose: **proving the code works**. For ablation science, use a GPU.
