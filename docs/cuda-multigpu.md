# DS4 Native CUDA Multi-GPU (Experimental)

Status: experimental, Linux/CUDA only. Developed and tested on 2× RTX 3090
(sm_86, 24 GB each) with CUDA 13.x, driver 580+.

## Problem

DS4's CUDA backend runs the whole model graph on one device. When the model is
larger than VRAM (DeepSeek V4 Flash IQ2XXS is ~81 GiB against 24 GB of VRAM),
the routed-expert weights are read from host memory at kernel time (zero-copy
over PCIe) or staged per token in `--ssd-streaming` mode. Either way decode is
bound by host→device weight traffic on a single PCIe link, and every other GPU
in the box stays idle.

## Measured starting point (2× RTX 3090, GPU1 idle)

```
./ds4 -m ds4flash.gguf --nothink -n 24 -p "..."   (ctx=32768)
prefill: 1.43 t/s   generation: 0.68 t/s          (default mode)
prefill: 0.69 t/s   generation: 0.88 t/s          (--ssd-streaming, 512-expert cache)
```

Per decoded token the router selects 6 of 256 experts in each of the 40 MoE
layers (43 layers, 3 hash-routed). At ~6.75 MiB per expert that is ~1.7 GiB of
expert weights touched per token. Whatever is not resident in VRAM must cross
PCIe: that, not compute, is the decode bottleneck.

## Why not layer split / tensor parallel first

This machine's GPUs are asymmetric: GPU0 has PCIe 4.0 x16 (~25 GB/s), GPU1
PCIe 4.0 x4 (~6 GB/s), no NVLink, no PCIe P2P (GeForce). A layer split makes
GPU1 stream the expert weights for its layers over a x4 link — it would be
4× slower than GPU0 at the exact thing that bounds decode. Dense tensor
parallelism splits matmuls that are already fast (the dense weights fit in
VRAM); it parallelizes the part that is not the bottleneck and adds an
all-reduce per matmul over a slow link.

For a MoE model the memory-capacity problem *is* the performance problem, so
the design goal is: maximize the number of expert weights resident across all
GPUs, never let the slow-link GPU stream weights at decode time, and keep all
activation traffic (KBs per layer) on the fast paths.

## Architecture: primary graph device + expert banks

```
DS4 process (single)
├── device 0 (primary, fast link)
│   ├── whole model graph: attention, KV, router, shared experts, norms, output
│   ├── q8→f16 dense-weight cache, streaming expert LRU cache (existing code)
│   └── streams misses over its x16 link (existing --ssd-streaming machinery)
└── device 1..N (secondary)
    ├── static "expert bank": as many routed experts as VRAM allows,
    │   loaded once (hotlist seed + lazy first-touch fill), never evicted,
    │   never streamed at decode time
    └── computes the routed-expert FFN for the experts it owns
```

The graph itself is not split. Per MoE layer, the selected (token, expert)
pairs are partitioned by ownership:

1. `ds4_gpu_stream_expert_cache_begin_selected_load` (decode) and
   `..._prepare_selected_batch` (prefill) already see the selected expert ids
   on the host. There, each pair is tested against the secondary banks.
   Bank-owned pairs are excluded from the primary staging load and recorded in
   a pending per-layer partition.
2. `ds4_gpu_routed_moe_{one,batch}_tensor` runs the *same* MoE launch path on
   every device with complementary weight masks:
   - the router weight of a pair is kept only on the device that owns the
     expert; on every other device that pair gets weight 0 and its expert id
     is remapped to a resident slot (id 0). Since the MoE kernels apply the
     router weight linearly (`mid = swiglu(gate,up) · w`), a zero-weight pair
     contributes exactly 0 and costs only a few µs of redundant VRAM reads.
   - the token activations are quantized to q8_K once on the primary and the
     compact q8 activations (~4.7 KiB/token) are copied to the secondaries.
   - each secondary writes its partial `routed_out` (n_tokens × 4096 f32),
     which is copied back and added into the primary's `routed_out`.
3. All cross-device traffic uses pinned bounce buffers with explicit CUDA
   events (D2H on the source stream → H2D on the destination stream). No P2P
   requirement; if CUDA peer access is available it is used directly.

This keeps the existing single-GPU code path byte-identical (the partition
step is a no-op when no secondary device is configured), reuses every existing
MoE kernel unchanged, and matches the PCIe asymmetry: the x4 GPU only ever
receives ~5 KiB/token/layer of activations and sends 16 KiB/token/layer of
partial outputs.

Expected VRAM budget on this machine: ~19–21 GiB of bank on GPU1 (~2 900
experts of 11 008) plus the existing primary-side caches — roughly 2.5× the
resident expert coverage of a single GPU, with both GPUs computing expert
FFNs in parallel.

## CLI

```
--cuda-devices <list|auto>     e.g. 0,1 — first is the primary/graph device
--cuda-split <off|auto|experts>  auto = experts for MoE models
--cuda-expert-bank <N>GB       per-secondary bank budget (default: free VRAM - reserve)
--cuda-p2p <auto|on|off>       peer access if the platform allows it
--cuda-print-topology          print device/link/P2P matrix at startup
```

`--cuda-split experts` implies `--ssd-streaming` (the partition points and the
compact-slot kernels live in that mode). Layer split and dense tensor
parallelism are intentionally not implemented; see "Why not" above.

## Files

- `ds4_cuda.cu` — device contexts, expert bank, partition + masked dispatch
- `ds4.c` — option plumbing, bank seeding alongside the hotlist seed
- `ds4_cli.c`, `ds4_bench.c`, `ds4_server.c` — flags

## Measured results (2× RTX 3090, x16 + x4, no NVLink)

`ds4-bench`, DeepSeek V4 Flash IQ2XXS (~81 GiB), `promessi_sposi.txt`,
`--ssd-streaming-cache-experts 8GB`, 64 generated tokens per point. The
secondary bank auto-sized to 3155 expert slots (20.8 GiB); bank fill takes a
few seconds with a warm page cache.

```text
ctx     prefill t/s          generation t/s
        1 GPU   2 GPU        1 GPU   2 GPU
2048    34.8    48.6 (+40%)  0.74    0.97 (+31%)
4096    33.6    48.1 (+43%)  0.74    0.96 (+30%)
6144    32.8    47.2 (+44%)  0.73    0.93 (+27%)
8192    32.9    45.6 (+39%)  0.74    0.93 (+26%)
```

Short-prompt decode with a warm primary expert cache: 1.80 → 2.82 t/s
(+57%). Original pre-patch default-mode baseline on the same machine and
prompt: 1.43 t/s prefill, 0.68 t/s generation.

Correctness: greedy (`--temp 0`) output is token-for-token identical between
single-GPU and multi-GPU runs of the same prompt.

Benchmark logs: `/models/ds4/logs/bench-single-gpu0-2k-8k.txt`,
`/models/ds4/logs/bench-native-multigpu-2k-8k.txt` (machine-local, not
committed).

## Known limitations

- The secondary devices accelerate only the routed-expert FFN. Attention,
  shared experts and dense projections stay on the primary.
- No automatic VRAM pooling: a 48 GB pair is not a 48 GB GPU.
- Startup pays one pass to fill the banks (bounded by SSD/page-cache speed).
- GeForce PCIe P2P is disabled by NVIDIA; transfers go through pinned host
  memory. NVLink, if present, is picked up automatically via peer access.
- Zero-weight (masked) pairs are still computed on every device — the kernels
  apply the router weight after the expert matmuls. Decode waste is
  negligible (a handful of pairs); in large prefill batches each device does
  the full pair count of MoE FLOPs, so the prefill win comes from reduced
  staging, not reduced compute. Future work: skip zero-weight pairs when
  building the sorted-pair buckets (needs coordinated handling in the
  non-atomic down-gather, which reads the per-pair down buffer).
- The primary's streaming expert LRU cache is released by every prefill
  chunk (upstream behavior of the batch staging path), so decode right after
  a long prefill starts cold. Future work: preserve or re-seed it.
