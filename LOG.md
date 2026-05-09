# Ablation Mechanism Log

## 2026-05-09: CPU smoke test (all mechanisms)

**Setup**: depth=3, dim=192, heads=3, head_dim=64, seq=128, batch=4, vocab=8192, Adam lr=1e-3, 30s per config, CPU (no GPU), SDPA instead of FA3.

| Config | Train loss | Val loss | Steps | Params |
|--------|-----------|----------|-------|--------|
| Baseline (3 content) | 7.204 | 7.551 | 212 | 7,618,758 |
| 5c + 1 orig RRPRAM | 6.926 | 7.530 | 218 | 6,570,118 |
| 5c + 1 RoPE-RRPRAM (shared v) | 7.714 | 7.557 | 211 | 6,533,254 |
| 5c + 1 RoPE-RRPRAM (sep v) | 7.218 | 7.511 | 214 | 6,570,118 |
| 5c + 1 Janus | 6.853 | 7.479 | 221 | 6,533,254 |

**Result**: All 5 configs train stably. No NaN, all gradients flow. Loss differences are not meaningful at this scale (see analysis below).
