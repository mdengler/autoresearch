"""GPU ablation test: runs all mechanism configs via train.py and compares results."""
import os
import re
import subprocess
import sys


DEPTH = int(os.environ.get("DEPTH", 12))
SEED = int(os.environ.get("SEED", 42))
DEVICE_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", 128))

CONFIGS = [
    ("baseline",                    {}),
    ("5c + 1 Janus",                {"N_JANUS": "1"}),
    ("5c + 1 RoPE-RRPRAM (shared)", {"N_RRPRAM": "1", "RRPRAM_SHARED_V": "1"}),
    ("5c + 1 RoPE-RRPRAM (sep v)",  {"N_RRPRAM": "1", "RRPRAM_SHARED_V": "0"}),
    ("5c + 1 orig RRPRAM",          {"N_RRPRAM_ORIGINAL": "1", "DEVICE_BATCH_SIZE": "32"}),
]

SUMMARY_FIELDS = [
    "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
    "mfu_percent", "tok_per_sec", "total_tokens_M", "num_steps",
    "num_params_M", "depth", "seed",
]


def parse_summary(output):
    """Extract key: value pairs from the --- summary block at end of train.py output."""
    metrics = {}
    in_summary = False
    for line in output.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if not in_summary:
            continue
        m = re.match(r"^(\w+):\s+(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def run_config(name, extra_env):
    env = os.environ.copy()
    env["DEPTH"] = str(DEPTH)
    env["SEED"] = str(SEED)
    env["DEVICE_BATCH_SIZE"] = str(DEVICE_BATCH_SIZE)
    # Reset mechanism vars to zero so previous config doesn't leak
    for var in ("N_JANUS", "N_RRPRAM", "N_RRPRAM_ORIGINAL", "RRPRAM_SHARED_V", "USE_MECH_GATE"):
        env.pop(var, None)
    env.update(extra_env)

    cmd = ["uv", "run", "train.py"]
    cmd_display = " ".join(f"{k}={v}" for k, v in sorted(extra_env.items()))
    if cmd_display:
        cmd_display = f"DEPTH={DEPTH} SEED={SEED} {cmd_display} uv run train.py"
    else:
        cmd_display = f"DEPTH={DEPTH} SEED={SEED} uv run train.py"

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Config: {name}", file=sys.stderr)
    print(f"  {cmd_display}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s", file=sys.stderr)
        return False, {}

    # Stream output to stderr for visibility
    if result.stdout:
        for line in result.stdout.splitlines()[-20:]:
            print(f"  {line}", file=sys.stderr)
    if result.returncode != 0:
        print(f"  EXIT CODE {result.returncode}", file=sys.stderr)
        if result.stderr:
            for line in result.stderr.splitlines()[-10:]:
                print(f"  stderr: {line}", file=sys.stderr)
        return False, {}

    metrics = parse_summary(result.stdout)
    if not metrics:
        print(f"  WARNING: no summary block found in output", file=sys.stderr)
        return False, {}

    print(f"  val_bpb={metrics.get('val_bpb', '?')}  tok/sec={metrics.get('tok_per_sec', '?')}", file=sys.stderr)
    return True, metrics


def main():
    results = {}
    for name, extra_env in CONFIGS:
        ok, metrics = run_config(name, extra_env)
        results[name] = (ok, metrics)

    # Summary table
    hdr = (
        f"{'Config':40s}  {'Pass':4s}  {'val_bpb':>8s}  {'tok/sec':>10s}"
        f"  {'tokens_M':>8s}  {'Steps':>5s}  {'VRAM_MB':>8s}  {'Params_M':>8s}"
    )
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print(hdr, file=sys.stderr)
    for name, (ok, m) in results.items():
        if ok and m:
            tok_sec = int(m.get("tok_per_sec", 0))
            print(
                f"  {name:38s}  {'PASS':4s}"
                f"  {m.get('val_bpb', 0):8.4f}"
                f"  {tok_sec:>10,}"
                f"  {m.get('total_tokens_M', 0):8.1f}"
                f"  {int(m.get('num_steps', 0)):5d}"
                f"  {m.get('peak_vram_mb', 0):8.1f}"
                f"  {m.get('num_params_M', 0):8.1f}",
                file=sys.stderr,
            )
        else:
            print(f"  {name:38s}  FAIL", file=sys.stderr)

    # Machine-readable TSV to stdout
    tsv_cols = ["config", "status", "val_bpb", "tok_per_sec", "total_tokens_M",
                "num_steps", "peak_vram_mb", "num_params_M", "training_seconds", "seed"]
    print("\t".join(tsv_cols))
    for name, (ok, m) in results.items():
        if ok and m:
            row = [
                name, "pass",
                f"{m.get('val_bpb', 0):.6f}",
                f"{int(m.get('tok_per_sec', 0))}",
                f"{m.get('total_tokens_M', 0):.1f}",
                f"{int(m.get('num_steps', 0))}",
                f"{m.get('peak_vram_mb', 0):.1f}",
                f"{m.get('num_params_M', 0):.1f}",
                f"{m.get('training_seconds', 0):.1f}",
                f"{int(m.get('seed', 0))}",
            ]
        else:
            row = [name, "fail"] + [""] * (len(tsv_cols) - 2)
        print("\t".join(row))


if __name__ == "__main__":
    main()
