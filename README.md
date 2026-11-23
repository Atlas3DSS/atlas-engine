# Atlas Engine

MIT-licensed CUDA inference engine for **Inkling-2-Mini** (built on [`llama.cpp`](docs/UPSTREAM_LLAMA_CPP_README.md) and [`realugbun/llama.cpp:inkling2-speed`](https://github.com/realugbun/llama.cpp)), with a reference **verified ledger** continual-memory implementation in [`ledger/verified_ledger.py`](ledger/verified_ledger.py).

## Changes over upstream

All five changes are submitted upstream to `realugbun/llama.cpp`:

1. **On-the-fly relative-bias FlashAttention and FP8/Hybrid KV** ([PR #2](https://github.com/realugbun/llama.cpp/pull/2)): Computes Inkling-2 relative-position bias inside MMA FlashAttention (`fattn-mma-f16-rel.cu`) without an $O(N^2)$ mask tensor. Supports direct `F8_E4M3`, `Q8_0`, and `F16` KV reads, odd-GQA `64x1` prefill dispatch (`gqa_ratio = 3`), `warp_all_far` fast path (`0.000000` max abs diff vs general path on SM86/SM89/SM120), and `ATLAS_HYBRID_SWA_F16=1` (`F16` sliding-window layers, `F8_E4M3` full-attention layers). Requires NVIDIA SM75+; older GPUs use the legacy path.
2. **Opt-in FP32 attention accumulation** ([PR #5](https://github.com/realugbun/llama.cpp/pull/5)): `ATLAS_RELBIAS_FP32_ACC=1` (default off) accumulates `P * V` in `float` instead of `half2`, cutting 196K per-layer drift vs `FP64` from `0.37%-1.38%` to `0.027%-0.035%` (`1-3%` prefill and `0.07-0.18%` decode overhead on RTX 3090).
3. **DSpark `f_rnn` ring rollback** ([PR #3](https://github.com/realugbun/llama.cpp/pull/3)): Per-sequence `dspark_f_rnn_dev` ring buffer (`ring_sz = 8`) keeps partial speculative draft rejections from corrupting `f_rnn` state.
4. **Per-request `loop_passes` override** ([PR #4](https://github.com/realugbun/llama.cpp/pull/4)): Sets `"loop_passes"` per request (`0` = default, `1..loop.count`) with slot-level KV isolation and mixed-depth batching.
5. **Opt-in `loop_passes` router** ([PR #6](https://github.com/realugbun/llama.cpp/pull/6)): `"loop_passes": -1` or `ATLAS_LOOP_ROUTER=1` (default off) assigns 1, 2, or 3 passes before slot launch. Intended for variable-loop weights only; stock fixed-3-pass weights degrade at 1-2 passes (`2.3x-2.4x` PPL).

## 196K benchmarks

Four cold long-document prompts (`112K-195K` tokens, 3 repeats each = 12 runs, full 3-pass depth, plus multi-slot KV-hole check W5):

| GPU / Setup | `-ub` | `token_embd` | Exact Answers | Prefill (tok/s) | Decode (tok/s) | Mean Total (s) | Peak VRAM |
|---|---:|---|---:|---:|---:|---:|---:|
| RTX 3090 (SM86) — Atlas Engine + DSpark (`IQ4_NL`) | 512 | CPU | 12/12 + W5 | `1,697.3 ± 9.1` | `86.1 ± 1.9` | `106.31 ± 1.05` | `22,456 MiB` |
| RTX 3090 (SM86) — Atlas Engine + DSpark (`IQ4_NL`) | 1024 | CPU | 12/12 | `1,764.6 ± 9.7` | `88.5 ± 2.0` | `102.83 ± 1.13` | `23,698 MiB` |
| RTX 3090 (SM86) — Maya v1.1.73 (`UD-Q4_K_XL`, ref) | — | CPU | 12/12 | `1,631.2 ± 5.9` | `86.8 ± 0.0` | `109.76 ± 0.54` | `21,396 MiB` |
| RTX 4090 (SM89) — Atlas Engine + DSpark (`IQ4_NL`) | 512 | CPU | 12/12 + W5 | `3,253.4 ± 79.3` | `113.3 ± 1.4` | `57.44 ± 1.31` | `23,220 MiB` |
| RTX PRO 6000 Blackwell (SM120) — Atlas Engine + DSpark (`IQ4_NL`) | 512 | GPU | 12/12 + W5 | `4,465.3 ± 30.9` | `155.9 ± 3.4` | `41.97 ± 0.39` | `24,498 MiB` |

### Numerical audit

SM86, FP8 KV, non-repeating inputs, compared against an `FP64` reference:

- `warp_all_far` vs general path: `0.000000` max abs diff on SM86, SM89, and SM120.
- 196K per-layer drift (`n_q=16 | n_q=1`): legacy `half2` `0.708% | 0.907%`, Atlas Engine `half2` `1.377% | 0.374%`, Atlas Engine `ATLAS_RELBIAS_FP32_ACC=1` `0.0274% | 0.0346%` (`norm_ratio = 0.99991`).
- 64K unseen prose (`PPL ~19.2`, 256 tokens): all path differences are within `<1σ` of the `~0.010-0.015` routing KL floor (196K end-to-end not measured).

### Dynamic `loop_passes`

On variable-loop weights (`varloop-Q4_0`, RTX 4090, `-c 4096`, no DSpark):
- 3 passes: `106.9 tok/s`, `PPL 2.4627 ± 0.0708`
- 2 passes: `141.0 tok/s` (`1.32x`), `+2.9%` PPL (`94.06%` top-1)
- 1 pass: `200.0 tok/s` (`1.87x`), `+7.0%` PPL (`90.38%` top-1)

## Verified ledger and elastic GGUFs

[`ledger/verified_ledger.py`](ledger/verified_ledger.py) is a host-verified 4-bank continual-memory ledger (schema/JOINs, verified traces, error repairs, and schema-drift migration with drift invalidation).

Loop-tiered BF16-sourced GGUFs (`7.90-12.60 GiB`) and full eval JSONs are on Hugging Face: [`Atlas3D/Inkling-2-Mini-Atlas-Elastic-GGUF`](https://huggingface.co/Atlas3D/Inkling-2-Mini-Atlas-Elastic-GGUF).

CL-Bench `database_exploration` dev split (`--parallel 5`, Hybrid FP8 KV, 0 HTTP errors, 0% `no_answer`):

| Quant | Size / Peak 64K VRAM | 64-ch KLD vs BF16 | Bare Score / Acc | +Verified Ledger Score / Acc | Queries (Bare -> +Ledger) |
|---|---:|---:|---:|---:|---:|
| `Atlas-8G-Q20Down` | `7.90 / 11.33 GiB` | `0.3217` | `0.0203 / 19.0%` | `0.1793 / 33.5%` | `14.16 -> 8.69` |
| `Atlas-8G` | `9.68 / 13.11 GiB` | `0.0865` | `0.0650 / 35.5%` | `0.2877 / 39.5%` | `13.40 -> 4.58` |
| `Atlas-9G` (`n=10`) | `10.46 / 13.89 GiB` | `0.0567` | `0.0483 / 29.5%` | `0.2397 / 35.8%` | `12.34 -> 5.91` |
| `Atlas-12G` | `12.60 / 15.92 GiB` | `0.0199` | `0.0447 / 27.5%` | `0.3210 / 42.5%` | `13.12 -> 4.94` |
| `Q8_0` ref | `25.91 / 29.26 GiB` | `0.0032` | — | `0.3167 / 41.5%` | `4.17` |

Every Bare vs +Verified Ledger rung above separates completely at the run level (`p_run = 0.0079` for `n=5 vs 5`, `p_run = 0.0002` for `Atlas-9G` `n=10 vs 5`).

## Build and run

Requires NVIDIA SM75+ (Turing, Ampere, Ada, or Blackwell):

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --target llama-server llama-bench llama-perplexity -j
```

Serve Inkling-2-Mini at 196K context with Hybrid FP16-SWA / FP8-Full KV and DSpark:

```bash
export ATLAS_HYBRID_SWA_F16=1
# Optional: export ATLAS_RELBIAS_FP32_ACC=1
# Optional (variable-loop weights only): export ATLAS_LOOP_ROUTER=1

./build/bin/llama-server \
  -m Inkling-2-Mini-IQ4_NL.gguf \
  -md dspark-Inkling-2-Mini-Q8_0.gguf -ngld 999 \
  -ngl 999 -fa on \
  -ot "token_embd.weight=CPU" \
  -ctk f8_e4m3 -ctv f8_e4m3 \
  -b 2048 -ub 512 \
  -c 196608 --parallel 1 \
  --host 127.0.0.1 --port 8080
```

Omit `-ot "token_embd.weight=CPU"` on 32+ GB GPUs.

## License and credits

- MIT License ([LICENSE](LICENSE)).
- Built on [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) and [`realugbun/llama.cpp`](https://github.com/realugbun/llama.cpp) (`inkling2-speed`).
- Inkling-2-Mini by Thinking Machines Lab. Reference numbers for `project-maya` (`Maya v1.1.73`, Apache 2.0) were measured externally; no Maya kernel code is included.
