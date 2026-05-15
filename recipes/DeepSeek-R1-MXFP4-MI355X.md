# DeepSeek-R1 MXFP4 on MI355X

Operational recipe for serving `amd/DeepSeek-R1-0528-MXFP4` with MTP
speculative decoding on the AMD MI355X. Per-concurrency launch flags
and env-knob tuning notes that produced the measured throughput numbers
in the table below.

Full investigation artifacts (slot-by-slot trail, kernel-level work,
profile findings, full lever sweep with mechanisms, every config and
script used, raw bench JSONs) live in the working repo at
[`ihamzazam/amd-e2e-dsr1`](https://github.com/ihamzazam/amd-e2e-dsr1).

## Setup

Container, HF cache mount, GPU device flags, and `dsr1_benchmark`
compilation follow the bounty quickstart at
[`amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/COMPETITION_QUICKSTART_EN.md`](https://github.com/ROCm/amdgpu_bounty_optimization/blob/main/dsr1-fp4-atom-mtp-mi355x/COMPETITION_QUICKSTART_EN.md).
The launch flags below replace `launch_atom_server.sh` /
`specific_conc_var.sh`; everything else (image, bench binary, leaderboard
submit) is unchanged.

Pins used for the numbers below:

- ATOM — this fork at branch [`bounty-dsr1-mxfp4-mi355x`](https://github.com/ihamzazam/ATOM/tree/bounty-dsr1-mxfp4-mi355x) (commit `d372640a` + the env-var change in this PR).
- aiter — upstream [`ROCm/aiter`](https://github.com/ROCm/aiter) at commit `b8aacd39`.
- Image — `rocm/atom:rocm7.2.0-ubuntu24.04-pytorch2.9-atom0.1.1`.
- Model — `amd/DeepSeek-R1-0528-MXFP4` (canonical bounty model).
- Workload — ISL=8192, OSL=1024, MTP `k=3`, `kv_cache_dtype=fp8`.

On the host, after launching the container per the quickstart, clone
the pinned source trees and install editable inside the container:

```bash
# On host — replaces the ATOM + aiter clones from quickstart step 1
git clone --branch bounty-dsr1-mxfp4-mi355x https://github.com/ihamzazam/ATOM ~/competition/ATOM
git clone --recursive https://github.com/ROCm/aiter ~/competition/aiter
( cd ~/competition/aiter && git checkout b8aacd39 && git submodule update --init --recursive )

# Inside the container (mounts from quickstart step 2)
cd /workspace/aiter && python3 setup.py develop
cd /workspace/ATOM  && pip install -e .
```

The bench binary build (`g++ -std=c++17 -o dsr1_benchmark
dsr1_benchmark.cpp -lcurl -pthread -O2`) is unchanged from the
quickstart.

## Launching the server

The right TP and capture-size set differs by concurrency. Use one of
the three launches below for the matching CONC.

### CONC=4 (TP=4)

```bash
export ATOM_ENABLE_RELAXED_MTP=1
export ATOM_RELAXED_MTP_TOP_N=15
export ATOM_RELAXED_MTP_DELTA=0.30
export ATOM_USE_TRITON_MXFP4_BMM=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
export ATOM_DRAFTER_MANUAL_GRAPH=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export AITER_USE_NT=0
export TOPK_FORCE_PATH=one
export TOPK_DISPATCH_FACTOR=1
export AITER_AR_1STAGE=0
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --kv_cache_dtype fp8 -tp 4 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --max-num-batched-tokens 65536 --max-num-seqs 256 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,64,128]' \
  --gpu-memory-utilization 0.85
```

### CONC=32 (TP=4)

```bash
export ATOM_ENABLE_RELAXED_MTP=1
# upstream defaults for top/delta are fine at CONC=32
export ATOM_USE_TRITON_MXFP4_BMM=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export AITER_AR_1STAGE=0
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --kv_cache_dtype fp8 -tp 4 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --max-num-batched-tokens 32768 --max-num-seqs 1024 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32]'
```

### CONC=128 (TP=8)

```bash
export ATOM_ENABLE_RELAXED_MTP=1
export ATOM_USE_TRITON_MXFP4_BMM=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
export ATOM_DRAFTER_MANUAL_GRAPH=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_OPUS_MOE_SORTING=1
export AITER_AR_1STAGE=0
export AITER_LOG_LEVEL=WARNING
export MORI_SHMEM_HEAP_SIZE=8G

python -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --kv_cache_dtype fp8 -tp 8 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --max-num-batched-tokens 65536 --max-num-seqs 256 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,64,128]' \
  --gpu-memory-utilization 0.85
```

## Performance reference

Measured via the bounty's `dsr1_benchmark perf` harness on the pin and
flags above. `tput/GPU = total_token_throughput / (TP*DP)` per the
bench's own formula.

| ISL  | OSL  | CONC | TP | tput / GPU | median intvty | median e2e (ms) | GSM8K (flex-extract, 3-shot) |
| ---: | ---: | ---: | -: | ---------: | ------------: | --------------: | ---------------------------: |
| 8192 | 1024 | 4    | 4  | 1286       | 149           | 7171            | 0.9378                       |
| 8192 | 1024 | 32   | 4  | 3674       | 55.1          | 19559           | 0.9409                       |
| 8192 | 1024 | 128  | 8  | 4045       | 30.6          | 35660           | 0.9348                       |

## Notes on env-knob tuning for MXFP4 + MTP k=3

Behavior observed while validating the recipe. See
[`bounty/dsr1-mxfp4-mi355x/INVESTIGATION.md`](../bounty/dsr1-mxfp4-mi355x/INVESTIGATION.md)
for the full lever sweep with mechanisms.

### Helps

- `ATOM_RELAXED_MTP_TOP_N=15` / `ATOM_RELAXED_MTP_DELTA=0.30` at
  CONC=4. Upstream defaults (`10` / `0.6`) drop GSM8K below 0.93 on
  MXFP4 + MTP k=3; the tighter acceptance band keeps GSM8K above the
  gate while preserving most of the accept-rate gain. (Exposed via env
  in this same PR.)
- `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0` for small AR messages
  (CONC=4 / 32 with the bench shape above). The default cast wrapper
  is overhead-dominated at ~50 KB per AR; skipping it lowers per-AR
  latency. Stack with `AITER_USE_NT=0` + `TOPK_FORCE_PATH=one` at
  CONC=4 — alone it shifts GSM8K below 0.93 on MXFP4.
- `AITER_USE_OPUS_MOE_SORTING=1` at CONC=128 with TP=8 only. TP=4
  regresses with this flag.

### Avoid for MXFP4 + MTP k=3

- `ATOM_USE_UNIFIED_ATTN=1` at CONC=4. Numerics shift enough to drop
  GSM8K to ~0.927.
- `ATOM_USE_TRITON_MLA=1` with `--num-speculative-tokens 3`. The asm
  MLA decode kernel currently asserts `decode_qlen ∈ {2, 4}` for
  `gqa_ratio=32` in persistent mode; MTP k=3 produces qo_len=3 and
  SIGABRTs.
- Stacking ≥ 4 numerics-touching env vars at CONC=4. Cumulative drift
  pushed GSM8K below gate (914 stack qrcast + NT + TOPK + DRAFTER_CG
  = 0.9280).

### Concurrency / TP

- TP=8 at CONC=4 or CONC=32: total throughput is higher but
  `tput/GPU = total / (TP*DP)` drops the leaderboard tput below the
  small-CONC gate.
- TP=4 at CONC=128 on this pin: workers shut down at the GSM8K →
  bench-warmup transition (server still answers `/health`). TP=8 has
  enough per-GPU memory headroom to survive the warmup spike.

## Accuracy validation

Same approach as `recipes/DeepSeek-R1.md`, but with `--num_fewshot 3`
and `num_concurrent=65` to match the bounty harness:

```bash
lm_eval \
  --model local-completions \
  --model_args model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://localhost:8888/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```
