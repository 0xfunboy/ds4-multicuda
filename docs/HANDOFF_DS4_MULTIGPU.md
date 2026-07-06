# HANDOFF — DS4 Native CUDA Multi-GPU (feature/native-cuda-multigpu)

Ultimo aggiornamento: 2026-07-04 ~16:15 — LAVORO COMPLETO. Regression test OK (default single-GPU identico: 1.42/0.68 t/s). README+docs committati (8bd4858). Push fallito per auth GitHub mancante: l'utente deve eseguire `gh auth login` (o configurare un PAT) e poi `git push -u fork feature/native-cuda-multigpu`. Restano solo eventuali ottimizzazioni future (vedi Limitazioni).
Agente: Claude (Fable 5). Questo file permette a un altro agente di riprendere
il lavoro esattamente da dove è stato lasciato.

## Obiettivo del progetto
Patchare DS4 (`/home/cooper/ds4`, fork di antirez/ds4) perché usi ENTRAMBE le RTX 3090
in un solo processo e superi il baseline single-GPU. Brief completo in
`/home/cooper/FABLE5_DS4_MULTICUDA_PATCH_BRIEF.md`.

## STATO: implementazione COMPLETA e FUNZIONANTE, mancano solo docs/report finali

### Risultati misurati (modello DeepSeek V4 Flash IQ2XXS ~81GiB, ctx 32768 salvo indicato)
| Config | prefill t/s | gen t/s | note |
|---|---|---|---|
| Baseline default single GPU (pre-patch) | 1.43 | 0.68 | prompt corto, zero-copy PCIe |
| Single GPU --ssd-streaming 8GB | 1.07 | 1.80 | prompt corto, cache calda |
| **Multi-GPU 0,1 experts** | **1.80** | **2.82** | prompt corto, cache calda |
| Single GPU bench ctx2048 (promessi_sposi) | 38.13 | 0.74 | gen a cache fredda |
| **Multi-GPU bench ctx2048** | **46.71** | **0.94** | gen a cache fredda |
| Bench 2k-8k completo | vedi log | vedi log | /home/cooper/models/ds4/logs/bench-single-gpu0-2k-8k.txt e bench-native-multigpu-2k-8k.txt LETTI: prefill 34.8→48.6 (+40%), gen 0.74→0.97 (+31%) @2k; stabile fino a 8k |

**Correttezza verificata**: output greedy (--temp 0) IDENTICO token-per-token
tra single e multi GPU (48 token, stesso prompt).

### Architettura implementata (dettagli in /home/cooper/ds4/docs/cuda-multigpu.md)
- GPU0 (PCIe 4.0 x16) = primaria: tutto il grafo (attention, KV, router,
  shared experts, dense) + cache streaming esistenti + streaming dei miss.
- GPU1 (PCIe 4.0 **x4**!) = "expert bank" statica: 3155 slot esperti (20.8GiB)
  caricati una volta all'avvio (hotlist + riempimento round-robin), MAI
  ristreamati a runtime (il link x4 non regge lo streaming).
- Partizione delle coppie (token, esperto) per ownership nei punti di staging
  host-visibili (`begin_selected_load`/`prepare_selected_batch` in ds4_cuda.cu).
- Entrambe le GPU eseguono lo STESSO launch MoE su tutte le coppie con
  **maschere di peso complementari**: peso router vero solo sul device
  proprietario, 0 altrove (id rimappato a slot residente; i kernel clampano
  id<0 e applicano il peso linearmente post-swiglu ⇒ contributo esattamente 0).
- Parziali della secondaria raccolti via pinned memory + somma sulla primaria.
- Niente P2P richiesto (GeForce non lo ha su PCIe): staging pinned-host.
- Path single-GPU INVARIATO quando non c'è device list (ext=NULL ovunque).

### File modificati (tutti committati)
- `ds4_cuda.cu` — grosso: sezione multi-GPU (struct cuda_peer_device/bank,
  init/cleanup, pinned ring, partizione, routed_moe_launch con override
  sparsi `routed_moe_ext`, orchestratore `routed_moe_launch_multi`, kernel
  moe_mask_weights/moe_add_inplace, cuda_tmp_alloc_slot per-device).
- `ds4_gpu.h` — nuove API: ds4_gpu_set_cuda_devices/split/p2p/expert_bank_gb,
  ds4_gpu_cuda_multigpu_active, ds4_gpu_expert_bank_free_slots/fill_layer.
- `ds4.c` — seed banca da hotlist all'init sessione (2 punti: ds4_session e
  generate_metal_graph_raw_swa) + fill round-robin, plumbing opzioni engine,
  multi-GPU implica ssd_streaming.
- `ds4.h` — campi cuda_* in ds4_engine_options.
- `ds4_cli.c`, `ds4_bench.c`, `ds4_server.c` — flag --cuda-devices,
  --cuda-split, --cuda-p2p, --cuda-expert-bank.
- `ds4_help.c` — help dei nuovi flag.
- `docs/cuda-multigpu.md` — design document (aggiornare i numeri finali!).

### Commit (branch feature/native-cuda-multigpu, base = upstream main 80ebbc3)
```
683f7e3 cli: add --cuda-devices, --cuda-split, --cuda-p2p, --cuda-expert-bank
2aad734 core: seed CUDA expert banks and plumb multi-GPU engine options
be1b6dd cuda: add native in-process multi-GPU expert split
fa95fc9 docs: add native CUDA multi-GPU design document
```
Remote `fork` = https://github.com/0xfunboy/ds4-multicuda.git (già aggiunto).
**PUSH NON ANCORA FATTO: manca auth GitHub** (`gh auth login` mai eseguito).
git identity locale già configurata (0xfunboy / bisped@gmail.com).

## TODO RIMANENTI
Tutto fatto tranne:
1. PUSH: serve auth GitHub dell'utente (`gh auth login`), poi
   `cd /home/cooper/ds4 && git push -u fork feature/native-cuda-multigpu`.
2. (opzionale, futuro) ottimizzazioni in "Limitazioni note / futuro".

## Consigli hardware per l'utente (da includere nel report)
- **NVLink: NON necessario adesso.** Il design manda solo KB di attivazioni
  alla GPU1. Un bridge NVLink (3090 lo supporta) abiliterebbe P2P a ~56GB/s:
  utile SOLO se in futuro si implementa tensor-parallel o si stream-a la banca
  a runtime. Non è l'upgrade prioritario. SLI: irrilevante, non serve mai.
- **PRIORITÀ 1: spostare la GPU1 su uno slot x8/x16 CPU-attached se la scheda
  madre lo consente** (ora è a x4: LnkSta width x4 vs max x16). Con x8+ si può
  usare GPU1 anche per lo streaming dei miss ≈ raddoppio banda aggregata.
  Verificare con `sudo lspci -vv -s 04:00.0 | grep LnkSta` dopo lo spostamento.
- **PRIORITÀ 2: SSD NVMe dedicato per /home/cooper/models** (l'attuale Kingston NV2 è
  DRAM-less, 63% usurato, letture medie 168MB/s con P10=11MB/s). Un buon
  NVMe Gen4 (con DRAM, es. 990 Pro/SN850X) toglie i ~8 minuti di page-in a
  freddo e migliora i primi minuti di decode. Con 128GB di RAM il modello
  81GB sta in page cache, quindi conta solo per il cold start.
- **RAM**: 128GB bastano per Flash IQ2XXS (81GB). Per quant più grandi
  servirebbero 192GB+.

## Limitazioni note / futuro (da includere nel report e nei docs)
- Nel batch/prefill le coppie mascherate (peso 0) vengono comunque computate
  su entrambe le GPU (i kernel applicano il peso alla fine). Ottimizzazione
  futura: saltare le coppie a peso 0 in moe_count/scatter_sorted_pairs — ma
  attenzione al gather non-atomic che legge il buffer down per TUTTE le coppie
  (leggerebbe garbage): servono zeri o skip coerente in tutti i path.
- La cache LRU esperti della primaria viene svuotata ad ogni chunk di prefill
  (comportamento upstream: allow_global_cache=0 in prepare_selected_batch
  → release_all). Per questo la gen "post-prefill lungo" è più lenta della
  gen a cache calda. Fix futuro: preservare la cache o ri-seedarla.
- La riserva VRAM sulla primaria (11.78GiB) limita la sua cache esperti a
  ~881 slot; tuning possibile ma non fatto.
- Solo il FFN routed è multi-GPU; attention/shared/dense restano su GPU0.
- ds4-eval/ds4-agent non hanno i flag CLI (usare env DS4_CUDA_*).

## Gotchas operativi per l'agente
- La shell Bash resetta la cwd a /home/cooper: usare SEMPRE path assoluti
  o `cd /home/cooper/ds4 &&` nei comandi (specie background).
- Creare `/tmp/claude-1001/.../scratchpad` prima di usarlo (mkdir -p).
- Build: `export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" &&
  cd /home/cooper/ds4 && make cuda CUDA_ARCH=sm_86 -j20` (CUDA_ARCH=sm_86, NON 86).
- Run: serve `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
- Il modello: /home/cooper/ds4/ds4flash.gguf → /home/cooper/ds4/gguf/DeepSeek-V4-Flash-...gguf.
  NON committare `gguf` (symlink untracked) né log giganti.
- I due run ds4 non possono girare insieme (lock file + VRAM).
- Env di debug: DS4_CUDA_MULTIGPU_DEBUG (dichiarato, quasi inutilizzato).

## Testo README pronto (inserire prima di "## Distributed Inference")
Vedi la sezione già formattata nel messaggio precedente dell'agente, oppure
riscrivere brevemente: cos'è (multi-GPU nativo in-process, expert bank),
build, esempi ds4/ds4-bench con --cuda-devices 0,1 --cuda-split auto, flag,
numeri misurati, rimando a docs/cuda-multigpu.md, disclaimer no-VRAM-pooling.

---

# FASE 2 (2026-07-06): CONFRONTO DS4 vs llama.cpp vs beellama.cpp

Richiesta utente: testare lo stesso DeepSeek su llama.cpp (/home/cooper/llama.cpp)
e su beellama.cpp (fork con speculative decoding, clonato in /home/cooper/beellama.cpp),
più il modello Qwen3-Coder-Next già scaricato.

## Fatti accertati
- llama.cpp b9873 (a4107133a), build CUDA ok in /home/cooper/llama.cpp/build/bin.
- SUPPORTA sia `deepseek4` sia `qwen3next`.
- beellama.cpp: supporta `qwen3next` ma NON `deepseek4` (ha solo deepseek/2/2-ocr/32)
  → DeepSeek si confronta solo DS4 vs llama.cpp.
- beellama = fork llama.cpp con DFlash speculative, spec model-free (ngram/suffix),
  KV cache compressa (turbo2/3/4). Build: cmake con
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc (senza, fallisce con
  "ptxas fatal sm_52" su CUDA 13 + cmake 3.28).
- Modello Qwen: /home/cooper/models/qwen3-coder-next/unsloth-ud-q3-k-xl/
  Qwen3-Coder-Next-UD-Q3_K_XL.gguf (34GB, 80B-A3B MoE 512 esperti/10 usati).
- Macchina riavviata: path ora /home/cooper/ds4 e /home/cooper/models (non più /srv, /models).

## Risultati finora
| Test | pp2048 t/s | tg64 t/s |
|---|---|---|
| Qwen3-Coder-Next su llama.cpp 2 GPU (-fa 1, tutto in VRAM) | 2648.7 | 122.0 |
| DeepSeek V4 Flash su DS4 multi-GPU (rif. fase 1) | 48.6 | 0.97 (2.8 warm) |
| DeepSeek V4 Flash su llama.cpp -ncmoe 43 (esperti su CPU, 2 GPU) | 77.3 | **9.52** |
| DeepSeek V4 Flash su llama.cpp -ncmoe 21 | FALLITO (OOM silenzioso, ~48GB stimati) | - |
| DeepSeek V4 Flash su llama.cpp -ncmoe 25/31 default split | OOM su GPU1 (24.6GB richiesti: i layer pesanti finiscono tutti su device 1) | - |
| **DeepSeek V4 Flash su llama.cpp -ncmoe 31 -ts 33/10** | **102.4** | **11.10** |

CONCLUSIONE PARZIALE IMPORTANTE: llama.cpp con esperti calcolati su CPU
(--n-cpu-moe) batte DS4 di ~10x in generazione (9.52 vs 0.97) e supera anche
il suo prefill. Il collo di DS4 è lo streaming pesi via PCIe; llama.cpp evita
proprio il trasferimento calcolando gli esperti sulla CPU dalla RAM.

## In corso / TODO
1. [FATTO] DeepSeek su llama.cpp: migliore config trovata = -ncmoe 31 -ts 33/10
   (102.4 pp / 11.1 tg). Nota: con ncmoe<43 serve -ts manuale tipo 33/10 perché
   i layer con esperti in VRAM sono gli ultimi e il default li mette tutti su GPU1.
2. [in corso, task bbcgqm410] beellama Qwen: bench raw + llama-cli --spec-type
   none/suffix/copyspec/ngram-simple su prompt codice
   (/tmp/claude-1001/.../scratchpad/code_prompt.txt), --temp 0 -n 512.
3. Report comparativo finale (DS4 vs llama.cpp per DeepSeek; llama.cpp vs
   beellama per Qwen; conclusioni oneste su DS4 vs llama.cpp).


## ESITO FINALE FASE 2 (2026-07-06)
| Modello / Runtime | pp2048 t/s | tg t/s | Note |
|---|---|---|---|
| DeepSeek V4F @ DS4 multi-GPU (nostra patch) | 48.6 | 0.97 (2.8 warm) | streaming esperti via PCIe |
| DeepSeek V4F @ llama.cpp -ncmoe 43 | 77.3 | 9.52 | esperti su CPU |
| DeepSeek V4F @ llama.cpp -ncmoe 31 -ts 33/10 | 102.4 | 11.10 | MIGLIORE. Con ncmoe<43 serve -ts manuale |
| Qwen3-Coder-Next @ llama.cpp 2 GPU | 2648.7 | 122.0 | tutto in VRAM |
| Qwen3-Coder-Next @ beellama 2 GPU | 2342.0 | 116.6 (119.3 in cli) | pari a llama.cpp |
| Qwen3-Coder-Next @ beellama --spec-type suffix/copyspec/ngram | CRASH | CRASH | "failed to remove sequence" common.cpp:1497: qwen3next è ibrido-ricorrente, KV non riavvolgibile → speculative incompatibile col modello |

Note operative beellama: llama-cli non supporta -no-cnv (usare -st single-turn
o llama-completion); build richiede -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc.

CONCLUSIONE: per DeepSeek su questo hardware llama.cpp con esperti su CPU è
~10x più veloce in generazione del DS4 patchato (11.1 vs 0.97-2.8) e 2x in
prefill. La lezione: per MoE > VRAM conviene calcolare gli esperti sulla CPU
(RAM bandwidth) invece di trasferirli alla GPU. Un eventuale "DS4 fase 3"
dovrebbe implementare un percorso CPU per gli esperti mancanti dalla VRAM.
Per lavoro quotidiano di coding: Qwen3-Coder-Next su llama.cpp (122 t/s).
TUTTO COMPLETATO — resta solo il push del branch DS4 (serve gh auth login).

---

# FASE 3 (2026-07-06): Hybrid CPU-MoE backend

## Implementato (build OK, funziona)
- `--cuda-split hybrid`: banche GPU tengono gli esperti caldi, gli esperti freddi
  (non-bank) sono calcolati su CPU dai kernel quantizzati DS4 (no streaming PCIe).
- `--cpu-moe N` (primi N layer MoE forzati su CPU), `--cpu-moe-threads N`,
  `--cuda-hot-experts NGB` (alias di --cuda-expert-bank). Env: DS4_CPU_MOE_LAYERS.
- ds4.c: `ds4_cpu_moe_compute_hybrid` (riusa i worker batch iq2_xxs/q2_K).
- ds4_cuda.cu: partizione bank/CPU in `cuda_stream_selected_partition_and_load`,
  orchestratore `routed_moe_launch_multi` con pass CPU concorrente + somma parziale.
- File: ds4.c, ds4_cuda.cu, ds4_gpu.h, ds4.h, ds4_cli.c, ds4_bench.c, ds4_server.c, ds4_help.c.
- NON ancora committato.

## Risultati (short prompt, cache calda, ctx 32768)
| Config | gen t/s |
|---|---|
| baseline (fase 1) | 0.68 |
| experts (fase 2, stream cold) | 2.82 |
| **hybrid (fase 3, cold su CPU)** | **3.22** |
| all-CPU DS4 (--cpu-moe 43) | 2.90 |
| llama.cpp -ncmoe 31 -ts 33/10 | 11.1 |

## SCOPERTA CHIAVE (root cause del gap con llama.cpp)
I kernel CPU quantizzati di DS4 (ds4_vec_dot_q2_K_q8_K, ds4_vec_dot_iq2_xxs_q8_K)
hanno SOLO percorso ARM NEON (target Apple) + fallback SCALARE per x86.
CPU = i5-13500 (AVX2/FMA/F16C, no AVX-512). DS4 all-CPU=2.90 vs llama.cpp all-CPU=9.5
→ i kernel scalari x86 sono ~3x più lenti degli AVX2 a mano di llama.cpp.
L'hybrid è architetturalmente corretto ma il tetto è la velocità CPU x86.
Per il target ≥11 t/s serve: scrivere kernel AVX2 per iq2_xxs_q8_K e q2_K_q8_K
(riferimento in llama.cpp ggml-cpu), altrimenti il decode DS4 resta ~3 t/s.

## TODO
1. Profilo decode (in corso): capire se routed_moe domina il tempo (giustifica AVX2)
   o se il grafo GPU è il floor.
2. Se MoE domina → kernel AVX2 per i due dot product (blocco DS4: block_iq2_xxs, block_q2_K, block_q8_K).
3. Bench cold-cache ctx2048 experts vs hybrid.
4. Correttezza greedy --temp 0 (script scratchpad/verify_greedy.sh).
5. Docs + commit.

---

# FASE 3 AGGIORNAMENTO (2026-07-06 sera): kernel AVX2 + ottimizzazioni

## Risultati progressivi (decode t/s, DeepSeek V4 Flash, warm salvo nota)
| Step | decode t/s |
|---|---|
| baseline (fase 1) | 0.68 |
| experts (fase 2) | 2.82 |
| hybrid scalare (fase 3 iniziale) | 3.22 |
| + AVX2 (tabella 256KB) | 5.56 |
| + AVX2 L1-grid (2KB+1KB) | 5.56→7.84(20thr,-n96) |
| + skip-xq no-bank + auto-threads | 8.74 |
| + scheduler dinamico work-stealing | **9.83** |
| llama.cpp -ncmoe31 riferimento | 11.1 |

Cold bench ctx2048 (promessi_sposi): hybrid decode 6.83 (vs experts 1.03),
ma hybrid prefill era 21.2 (vs experts 53.7) → FIX: prefill batch grande usa
GPU streaming, decode usa CPU (allow_cpu only n_tokens<=8). Re-bench in corso.

## Cosa e' stato aggiunto (Fase 3, oltre l'hybrid backend)
1. Kernel AVX2 in ds4.c per x86 (prima solo NEON+scalare):
   - ds4_vec_dot_iq2_xxs_q8_K (L1-friendly: iq2xxs_grid 2KB + iq2xxs_signs 1KB,
     stile ggml, NO tabella signed 256KB che thrashava L2)
   - ds4_vec_dot_iq2_xxs_pair_q8_K (fusa, condivide load q8)
   - ds4_vec_dot_q2_K_q8_K ( port ggml AVX2 + helper ds4_hsum_float_8,
     ds4_scale_shuffle_q3k, DS4_MM256_SET_M128I)
   - guardia DS4_MOE_SCALAR per build di verifica scalare
2. Scheduler dinamico work-stealing nel thread pool (ds4_pool_drain + next_row/chunk
   atomici) - risolve gli straggler P/E-core dell'i5-13500. +12.5%. Migliora TUTTO il compute CPU.
3. ds4_cpu_moe_compute_hybrid: buffer persistenti (no malloc/free per-layer).
4. Orchestratore: skip quantize+D2H GPU quando non ci sono coppie bank.
5. Default cpu-moe-threads = nproc in hybrid.
6. Prefill/decode split: cold experts -> CPU solo per decode (n_tokens<=8),
   prefill grande -> GPU streaming.

## Correttezza VERIFICATA
- AVX2 vs scalare (build /tmp/ds4-scalar con -DDS4_MOE_SCALAR): token IDENTICI
  (I, primi, cin, que...); divergenza solo dopo ~18 token su un near-tie =
  riassociazione FP (i dot interi int8xint8 sono esatti). NON e' un bug.
- logprobs step 0-3 identici tra scalare e AVX2.

## TODO rimanenti
1. Confermare re-bench hybrid split (prefill deve tornare ~53).
2. (opz.) GPU0 hot bank: la cache 8GB su GPU0 e' usata in prefill ora; in decode
   potrebbe fare esperti in parallelo alla CPU (idle). Lever per >11 ma complesso.
3. Commit tutto (NON ancora committato la Fase 3). Files: ds4.c, ds4_cuda.cu,
   ds4_gpu.h, ds4.h, ds4_cli.c, ds4_bench.c, ds4_server.c, ds4_help.c.
4. Docs README + cuda-multigpu.md.

---

# FASE 3 - NOTA CRITICA GOVERNOR (2026-07-06 tarda sera)

## Scoperta: la varianza del decode (5-10 t/s) e' il CPU FREQUENCY GOVERNOR
- /sys/.../scaling_governor = "powersave", max 4.8GHz ma gira a ~900MHz nei burst.
- Il decode e' bursty (calcola ~1ms, aspetta la GPU ~1ms per layer): il governor
  vede ~50% utilizzo e NON boosta. CPU fredda (55C) ma a 900MHz-1.2GHz.
- llama.cpp e' stabile a 9.5 perche' il suo threadpool fa SPIN (tiene la CPU calda).
- NON ho root: non posso settare governor=performance. sudo chiede password.
- CONSIGLIO UTENTE (il piu' importante per il decode): 
  `sudo cpupower frequency-set -g performance`  (o scrivere "performance" in
  /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor). Vale ~2x sul decode.

## Spin-wait pool: PROVATO, disabilitato di default
- Ho aggiunto spin-wait bounded nel pool (g_pool_spin, env DS4_POOL_SPIN).
- A/B su questo box: spin PEGGIORA (4.00 vs 5.28 no-spin) perche' 19 core che
  spinnano spartiscono il power budget e i P-core boostano meno (powersave).
- Default g_pool_spin=0 (off). Su box con governor=performance potrebbe aiutare.

## Numeri onesti (governor powersave, senza root)
- prefill ctx2048: 54 t/s (come experts mode - lo split prefill=GPU funziona)
- decode: ~5 t/s cold / fino a ~9.8 quando la freq e' salita (sostenuto/warm)
- Con governor=performance atteso ~10-11 stabile (da confermare quando l'utente
  puo' settarlo).
- Correttezza: OK (AVX2==scalare token-identici).

## Config finale consigliata
./ds4 -m ds4flash.gguf --cuda-devices 0,1 --cuda-split hybrid \
  --ssd-streaming-cache-experts 8GB [--cpu-moe-threads 20 e' auto=nproc]
(+ governor=performance per il massimo)

---

# FASE 3 COMPLETATA (2026-07-06 notte)

## FIX determinismo (era un bug!)
- Hybrid era non-deterministico run-to-run (single-GPU invece SI'). Causa:
  il buffer device g_hybrid_part (parziale CPU) veniva riusato ogni layer
  senza aspettare l'add del layer precedente -> lettura stale timing-dependent.
- FIX: cudaStreamWaitEvent(xfer, g_peer_primary_add_ev) prima dell'H2D del
  parziale CPU (come fa gia' il gather delle banche). Ora f1==f2==f3 identici.

## Stato finale VERIFICATO
- Correttezza: hybrid DETERMINISTICO + AVX2==scalare (token-identici).
- prefill ctx2048: ~54 t/s | decode: ~6.6 sostenuti / ~9.8 warm (powersave governor)
- Tutto builda pulito (make cuda CUDA_ARCH=sm_86).

## Da fare: COMMIT (non ancora fatto) + push
File Fase 3: ds4.c ds4_cuda.cu ds4_gpu.h ds4.h ds4_cli.c ds4_bench.c
ds4_server.c ds4_help.c docs/cuda-multigpu.md README.md

## COMMIT FATTI (branch feature/hybrid-cpu-moe)
296bd11 docs: Phase 3 hybrid CPU-MoE
cac3352 cli: --cuda-split hybrid, --cpu-moe, --cpu-moe-threads, --cuda-hot-experts
a30adf7 cuda: hybrid split — cold experts on CPU, deterministic partial gather
1637086 cpu: AVX2 routed-expert kernels, work-stealing pool, hybrid CPU-MoE compute
(base: aeed188 su main)
PUSH: serve gh auth login, poi git push -u origin feature/hybrid-cpu-moe (o fork).
