# DS4 — LOG TEST DETTAGLIATO (cronologico)

Questo file è il registro completo di OGNI test, con comando esatto, parametri e
risultato. Va aggiornato ad ogni nuovo test (append in fondo) così nulla va perso
in caso di interruzione. Companion di `HANDOFF_DS4_MULTIGPU.md` (che ha il piano
e lo stato architetturale).

Hardware fisso: 2× RTX 3090 24GB (GPU0 PCIe 4.0 x16, GPU1 x4, no NVLink),
i5-13500 (6 P-core + 8 E-core = 20 thread, AVX2/FMA/F16C, no AVX-512), 128GB DDR5.
Modello fisso: `/home/cooper/models/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
(~81 GiB, arch deepseek4, 43 layer, 40 MoE, 256 esperti/layer, 6 usati).
Env build: `CUDA_HOME=/usr/local/cuda`, `make cuda CUDA_ARCH=sm_86 -j20`.
Env run: `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
`ds4flash.gguf` = symlink al modello.

Legenda: pp = prefill tokens/s, tg = generation tokens/s.

---

## FASE 1 — Baseline (DS4 upstream, prima delle patch)

| # | Comando (parametri) | pp t/s | tg t/s | Note |
|---|---|---|---|---|
| 1 | `./ds4 -m ds4flash.gguf --nothink -n 128 -p "..."` (ctx 32768, 1 GPU default) | 1.43 | 0.68 | GPU1 idle, streaming pesi via PCIe. È il baseline da battere. |

## FASE 2 — Multi-GPU "experts" (banca GPU1, streaming esperti freddi)

Config: `--cuda-devices 0,1 --cuda-split experts --ssd-streaming-cache-experts 8GB`.
Banca GPU1 auto = 3155 slot esperti (20.8 GiB). Riempita in ~2-3s.

| # | Comando/parametri | pp t/s | tg t/s | Note |
|---|---|---|---|---|
| 2 | `--cuda-split experts`, prompt corto, ctx 32768 (cache calda) | 1.80 | 2.82 | vs 0.68 baseline |
| 3 | single GPU `--ssd-streaming` 8GB, prompt corto | 1.07 | 1.80 | riferimento single |
| 4 | `ds4-bench ... experts`, promessi_sposi, ctx-start/max 2048, gen 64 (cold) | 53.70 | 1.03 | prefill GPU veloce, decode cold lento |
| 5 | bench 2k-8k experts (fase 1 confronto): pp 34.8/33.6/32.8/32.9, tg 0.74×4 | — | — | log /home/cooper/models/ds4/logs/ |

## Confronto llama.cpp / beellama (stesso hardware, riferimenti)

llama.cpp b9873 (a4107133a), CUDA. beellama fork (85e22ea).

| # | Runtime/config | pp t/s | tg t/s | Note |
|---|---|---|---|---|
| 6 | llama.cpp DeepSeek `-fa 1 -ncmoe 43` (tutti esperti su CPU, 2 GPU) | 77.3 | 9.52 | esperti su CPU/RAM |
| 7 | llama.cpp DeepSeek `-fa 1 -ncmoe 31 -ts 33/10` | **102.4** | **11.10** | MIGLIORE llama.cpp. ncmoe<43 richiede -ts manuale (senza va OOM su GPU1) |
| 8 | llama.cpp DeepSeek `-ncmoe 25/31` default split | OOM | — | i layer pesanti finiscono tutti su GPU1 |
| 9 | llama.cpp Qwen3-Coder-Next Q3_K_XL (34GB, tutto VRAM) `-fa 1 -p2048 -n64` | 2648.7 | 122.0 | modello che sta in VRAM |
| 10 | beellama Qwen3-Coder-Next stessa config | 2342.0 | 116.6 | pari a llama.cpp |
| 11 | beellama Qwen `--spec-type suffix/copyspec/ngram-simple` (codice) | CRASH | CRASH | "failed to remove sequence" common.cpp:1497 — qwen3next ibrido-ricorrente, KV non riavvolgibile → speculative incompatibile |

TARGET Fase 3: ≥ 11 t/s decode (= llama.cpp #7).

## FASE 3 — Hybrid CPU-MoE (esperti freddi su CPU, come llama.cpp --n-cpu-moe)

Config base: `--cuda-devices 0,1 --cuda-split hybrid --ssd-streaming-cache-experts 8GB`.
Nota: quasi tutti i test decode sono a prompt corto salvo indicato, ctx 32768.
**IMPORTANTE**: fino al 2026-07-06 sera il governor era `powersave` (CPU a
~900MHz-1GHz nei burst) → forte varianza. Dal 2026-07-06 notte governor gestito.

| # | Config / cosa testa | pp | tg | Note |
|---|---|---|---|---|
| 12 | hybrid, kernel CPU SCALARI (prima dell'AVX2), -n32 corto | 1.49 | 3.22 | primo hybrid funzionante |
| 13 | all-CPU scalare (`--cpu-moe 43`), -n32 | 1.61 | 2.90 | isola velocità CPU scalare: ~3x più lenta di llama.cpp (9.5) |
| 14 | profilo decode (DS4_METAL_DECODE_STAGE_PROFILE) | — | — | routed_moe = 4-11 ms/layer, tutto il resto <0.15ms → MoE = 95% del decode |
| 15 | hybrid + AVX2 (tabella signed 256KB), -n48 | 2.28 | 5.33 | AVX2 aiuta ma la tabella 256KB satura la L2 |
| 16 | all-CPU AVX2 (tabella 256KB), -n48 | 2.20 | 3.24 | tabella 256KB vanifica l'AVX2 (era 2.90 scalare) |
| 17 | hybrid + AVX2 **L1-grid** (2KB+1KB), -n48 | 2.28 | 5.56 | grid L1-friendly stile ggml |
| 18 | all-CPU AVX2 L1, sweep thread -n96: 12thr | 2.63 | 6.23 | |
| 19 | all-CPU AVX2 L1, 14thr -n96 | 2.75 | 6.45 | |
| 20 | all-CPU AVX2 L1, 20thr -n96 | 2.81 | 7.84 | 20 thread meglio (run lunghi = più stabili) |
| 21 | hybrid + skip-xq-no-bank + auto-thread, 20thr -n96 | 2.90 | 8.81 | banca GPU1 ora aiuta (+19% su all-CPU) |
| 22 | all-CPU (`--cpu-moe 43`), 20thr -n96 | 2.83 | 7.42 | |
| 23 | hybrid + fused-pair AVX2, auto-thread -n96 | 2.89 | 8.80 | fusione gate/up: q8 già in L1, invariato |
| 24 | hybrid + persistent-scratch, -n128 | 2.88 | 8.74 | malloc non era il collo |
| 25 | hybrid + **scheduler dinamico work-stealing**, -n128 | 3.02 | **9.83** | fix straggler P/E-core, +12.5% |
| 26 | bench cold ctx2048 experts vs hybrid: experts | 53.70 | 1.03 | |
| 27 | bench cold ctx2048 hybrid (prima dello split prefill/decode) | 21.21 | 6.83 | decode +6.6x ma prefill lento (CPU su batch 2048) |
| 28 | bench cold ctx2048 hybrid + **split prefill=GPU/decode=CPU** | 54.03 | 1.87 | prefill recuperato; tg basso = governor powersave (freq crollata) |
| 29 | diagnosi governor: CPU 55°C ma 877-1258 MHz durante decode | — | — | powersave + burst → no boost. Causa della varianza 3.5-9.8 |
| 30 | A/B spin-wait pool (-n200): SPIN | 1.86 | 4.00 | spin PEGGIORA su powersave (19 core spartiscono il power budget) |
| 31 | A/B: NO-SPIN (-n200) | 1.86 | 5.28 | spin disabilitato di default |
| 32 | run lungo -n400 (powersave, no-spin, freq media ~2GHz) | 2.12 | 6.59 | numero "sostenuto" su powersave |

### Correttezza (Fase 3)
- #C1 AVX2 vs SCALARE (build /tmp/ds4-scalar con -DDS4_MOE_SCALAR), all-CPU greedy --temp 0 -n64: token IDENTICI fino a char ~73 (~18 token), poi near-tie (riassociazione FP). Dot interi esatti.
- #C2 logprobs step 0-3 scalare vs AVX2: IDENTICI (I, primi, cin, que).
- #C3 single-GPU greedy run-to-run: DETERMINISTICO (identico).
- #C4 hybrid greedy run-to-run PRIMA del fix: NON deterministico (bug: g_hybrid_part riusato senza sync).
- #C5 hybrid 1-thread run-to-run: ancora NON det → non è il threading, è il pipeline del parziale.
- #C6 FIX (cudaStreamWaitEvent add_ev prima dell'H2D del parziale) → hybrid f1==f2==f3 DETERMINISTICO. ✓

### Stato dopo Fase 3 (governor powersave)
prefill ~54, decode ~6.6 sostenuti / ~9.8 picco warm. Correttezza OK.
Commit su feature/hybrid-cpu-moe: 1637086, a30adf7, cac3352, 296bd11.

---

## FASE 3b — Test con governor=performance (2026-07-06 notte, dopo che l'utente ha disabilitato il governor)

Governor ora `performance` (idle gia' a ~2993 MHz, boost a 4100 MHz sotto carico).

| # | Config / cosa testa | pp | tg | Note |
|---|---|---|---|---|
| 33 | hybrid -n256 (perf gov, freq 4100MHz nei sample decode) | 2.65 | 7.09 | governor NON era tutta la storia: a piena freq ancora 7.09 vs llama 11.1 |

| 34 | hybrid -n200 warm (perf gov) | 2.49 | 6.60 | |
| 35 | all-CPU -n200 warm (perf gov) | 2.46 | 6.44 | hybrid~=all-CPU: la banca GPU1 NON aiuta piu' a freq alta (overhead pipeline la cancella) |
| T | CPU a 82C sotto carico all-core AVX2 (soglia high 80C) | — | — | tetto termico i5-13500 65W |

ANALISI: DS4 all-CPU 6.44 vs llama.cpp all-CPU (#6) 9.5 = gap 1.47x nella CPU-MoE
(stesso confronto: GPU attention + tutti esperti CPU). Il gap NON e' la banca GPU,
e' il CPU-MoE + overhead per-layer. Prossimo: riprofilo per localizzare (kernel vs overhead).

| 36 | PROFILO decode perf gov (all-CPU): routed_moe 1.09ms best / 6.17ms worst, resto ~0.6ms/layer | — | — | routed_moe VARIA 6x per token (stesso lavoro!). best-case 1.09ms*40+26 = 70ms = 14 t/s. La varianza (dispatch latency thread dormienti + freq/thermal) uccide la media a 6.5 |

DIAGNOSI: il collo non e' il kernel (best-case 1.09ms/layer = veloce), e' la
VARIANZA per-token: i thread dormono tra dispatch e la wake latency (condvar)
varia. llama.cpp spinna -> dispatch istantaneo e costante -> 9.5-11. 
FIX in prova: spin-wait a budget moderato SU GOVERNOR PERFORMANCE (l'A/B #30-31
era su powersave dove lo spin scaldava troppo).

| # | Config | pp | tg | Note |
|---|---|---|---|---|
| 37 | spin sweep perf gov -n150: SPIN=0 | 2.20 | 5.72 | |
| 38 | SPIN=50000 | 2.32 | 5.18 | spin peggiora anche a perf gov (scalda->throttle) |
| 39 | SPIN=200000 | 2.32 | 4.58 | piu' spin = piu' lento. Spin CONFERMATO dannoso. |
| T2 | numeri calano 6.60->5.72 tra run: CPU calda 82C = THROTTLE | — | — | tetto termico e' ora il limite dominante a perf gov |

NUOVA DIAGNOSI: a governor performance il limite e' TERMICO (i5-13500 65W, all-core
AVX2 -> 82C -> throttle). Usare 20 thread (HT+E-core, inefficienti) fa piu' calore.
Test: MENO thread (P-core, piu' freddi/veloci sostenuti) = probabile cosa fa llama.cpp.
| 40 | thread sweep perf gov -n120 (CPU calda): 6thr | 2.48 | 5.13 | |
| 41 | 8thr | 2.41 | 5.41 | 8 thread = meglio a CPU calda |
| 42 | 12thr | 2.45 | 5.27 | |
| 43 | 16thr | 2.45 | 4.83 | piu' thread = piu' caldo = piu' lento |

CONFOUND TERMICO: tutte le misure ora sono sotto 6.60 perche' la CPU e' satura
dal testing continuo. I miei numeri llama.cpp (9.5/11.1) erano di STAMATTINA a CPU
FREDDA. Serve confronto EQUO stesse condizioni termiche -> rimisuro llama.cpp ORA.
| 44 | **llama.cpp -ncmoe31 -ts33/10 tg128 a CPU CALDA (96C, perf gov)** | 98.2 | **10.77** | llama.cpp e' TERMICAMENTE RESILIENTE: scende appena da 11.1 (cold) a 10.77 (hot 96C) |

CONCLUSIONE DIAGNOSTICA: il gap DS4(~5.5-6.6 hot) vs llama.cpp(10.77 hot) e' REALE
(~1.8x), non termico. llama.cpp sostiene ~10.8 sotto lo stesso calore; DS4 crolla.
llama.cpp e' strutturalmente piu' efficiente: quasi certamente CUDA GRAPHS (replay
del grafo decode, ~0 overhead di launch) mentre DS4 lancia ~1300 kernel/token
dall'host, interleavati col calcolo CPU MoE (il main thread fa sia launch CUDA
sia dot-product) -> jitter/serializzazione -> routed_moe varia 1-6ms/layer.
Il KERNEL CPU DS4 e' competitivo (best-case 1.09ms/layer = 14 t/s potenziale);
il problema e' l'overhead per-layer del pipeline decode, non il kernel.
| 45 | hybrid + blocking-sync event -n120 run1 | 2.70 | 5.00 | temp 72->80->100C durante il run |
| 46 | hybrid + blocking-sync run2 | 2.34 | 5.05 | blocking-sync NEUTRO (heat viene dal calcolo MoE, non dallo spin host). CPU tocca 100C = crit throttle |

CONCLUSIONE FINALE FASE 3b:
- Il gap DS4 vs llama.cpp e' OVERHEAD-BOUND, non CPU-compute-bound.
- DS4 CPU fa MENO expert-dot di llama.cpp (~160 vs 186/token, la banca offloada un po')
  ma e' piu' lento: l'overhead per-layer (D2H x, cudaEventSynchronize, dispatch pool,
  H2D partial, add, per 43 layer) tiene i core occupati piu' a lungo per token ->
  piu' calore -> throttle -> circolo vizioso.
- llama.cpp usa CUDA GRAPHS + ggml backend scheduler: ~0 overhead di launch, pipeline
  CPU/GPU efficiente -> sostiene 10.8 anche a 96C.
- Chiudere il gap richiede CUDA-graph capture del grafo decode DS4 (rework maggiore,
  fuori scope, alto rischio). Il KERNEL non e' il problema (gia' portato da ggml).
- Tetto termico i5-13500 (65W, 100C) limita comunque sia DS4 sia llama.cpp.

RISULTATO NETTO FASE 3: DS4 hybrid corretto+deterministico, 0.68 -> ~5-9.8 t/s
(9-14x baseline) secondo stato termico. Prefill ~54. Gap residuo vs llama (~1.8x)
= vantaggio infrastrutturale (CUDA graphs) di un engine maturo.
CONSIGLIO UTENTE: migliorare il raffreddamento CPU (o power-limit/undervolt) alza
il sostenuto di DS4 E llama.cpp; con meno thread (--cpu-moe-threads 8) DS4 throttla meno.
