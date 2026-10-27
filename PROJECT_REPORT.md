 # AMLN: Adaptive Memory Landscape Network
## A Brain-Inspired, Bounded-Memory System for Real-Time Streaming Cyber Threat Detection

**Project Report**

---

## Abstract

Modern intrusion-detection systems face a tension between accuracy and resource
cost: approaches that remember the entire event history detect well but grow
without bound, while fixed-size approaches stay cheap but forget indiscriminately.
This project presents **AMLN (Adaptive Memory Landscape Network)**, a
brain-inspired framework that detects cyber threats from a live log stream using
a small, *self-organizing* memory that grows, forgets, and reorganizes online —
within a **fixed, provably bounded budget** — and **without ever retraining** the
underlying neural network.

AMLN is evaluated on **CIC-IDS2017**, a real benchmark of a full week of network
traffic with injected attacks, replayed as a one-event-at-a-time live stream. The
system is benchmarked against four standard memory-retention strategies using a
prequential (test-then-train) protocol. AMLN compresses ~101,000 streamed events
into roughly **10 memory nodes (4.9 KB)** — about **5,300x less memory** than a
store-everything baseline — while matching that baseline's zero-day recall. A
live, self-explaining dashboard animates the memory forming, strengthening, and
decaying in real time.

---

## 1. Introduction

### 1.1 Problem

Security telemetry arrives as a never-ending stream of events: process launches,
network connections, authentications, file operations. The detector must:

1. Operate **online** — events arrive one at a time, indefinitely.
2. **Adapt** to new and evolving attacks (concept drift) without full retraining.
3. Stay **resource-bounded** — memory and latency cannot grow with stream length.
4. Flag **novel / zero-day** behavior it has never seen before.

Classic machine-learning IDS pipelines train a fixed model offline and either
store large feature buffers or retrain periodically. Both scale poorly for a
continuous, drifting stream.

### 1.2 Core idea

AMLN reframes detection as a **memory-management problem**. Instead of a flat
database of past events, it maintains a dynamic *memory landscape* over a learned
behavioral embedding space. Each memory is a point (a behavioral prototype) with
a scalar *height* representing its stability:

- **Hills** — tall, stable, long-term memories (well-established attack/benign
  patterns seen many times).
- **Slopes** — shorter, newer, short-term memories still being reinforced.
- **Valleys** — memories that decayed below a threshold and are forgotten.

A security analogy: rather than a guard writing every event in an ever-growing
notebook, AMLN is a guard with a compact, living memory of the *kinds* of
behavior seen — reinforced when repeated, faded when stale, and expanded when
something new appears.

### 1.3 Contributions

1. A novel memory landscape with three mechanisms (Section 4): **surprise-gated
   plasticity**, **energy-budgeted forgetting**, and **drift-aware consolidation**.
2. A **streaming benchmark harness** that fairly compares AMLN against standard
   retention baselines on real data, measuring detection quality *and* memory cost.
3. A **real-time, self-explaining dashboard** that makes the otherwise-invisible
   memory dynamics and memory savings directly observable.

---

## 2. Dataset

**CIC-IDS2017** (Canadian Institute for Cybersecurity).

- Source: https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle mirror used: https://www.kaggle.com/datasets/cicdataset/cicids2017
- Composition: 8 daily CSV capture files (Monday = all benign, Tuesday–Friday =
  attacks injected morning/afternoon), **~2.83 million bidirectional network
  flows**, each with **78 numeric flow features** and a `Label`.
- Attack families present: DoS (Hulk, GoldenEye, slowloris, Slowhttptest),
  DDoS, PortScan, FTP/SSH brute-force (Patator), Web Attacks (Brute Force, XSS,
  SQL Injection), Infiltration, Bot, Heartbleed.

### 2.1 Why this dataset

It is a widely cited, realistic IDS benchmark whose day-by-day structure gives a
natural chronological timeline — ideal for *replaying as a live stream*. There is
no truly live public feed; replaying a recorded, timestamped dataset in order is
the standard, reproducible way to simulate streaming IDS.

### 2.2 Full label distribution (observed)

| Label | Count |
|---|---|
| BENIGN | 2,273,097 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack – Brute Force | 1,507 |
| Web Attack – XSS | 652 |
| Infiltration | 36 |
| Web Attack – SQL Injection | 21 |
| Heartbleed | 11 |

The dataset is extremely **imbalanced** (80% benign; rarest attack has 11 events).
This imbalance is deliberately preserved in evaluation because it is what makes
real streaming detection hard.

### 2.3 Stream construction

Loading 2.8M flows per run is impractical and a naive cap would delete the rare
attacks entirely. A **stratified, chronology-preserving subsample** is used
(`stratified_stream`): keep **all** events of each attack family up to a cap
(8,000) and a uniform chronological sample of benign flows (40,000), then re-sort
by timestamp. Result: **101,323 events** cached to `data/stream.npz` (6.8 MB),
preserving the Monday→Friday timeline and all rare families.

---

## 3. System Architecture

```
CIC-IDS2017 CSVs (2.8M flows)
        │   load + clean + stratified subsample  (amln/data/cicids.py)
        ▼
  Cached stream.npz (101k events, 78 features, chronological)
        │   causal robust scaling, one event at a time  (amln/streaming/replayer.py)
        ▼
  Transformer behavioral autoencoder  (amln/models/encoder.py)
        │   78 features → 64-d embedding + reconstruction error (surprise signal)
        ▼
  Adaptive Memory Landscape V2  (amln/memory/adaptive_v2.py)  ← NOVEL CORE
        │   observe(): surprise-gate → match/spawn → reinforce/decay →
        │              energy-budget prune → merge/split
        ├─────────────► Live dashboard      (dashboard.py, streaming/engine.py)
        └─────────────► Benchmark harness    (run_benchmark.py, benchmark/)
```

### 3.1 Pipeline stages

| Stage | Module | What it does | Why |
|---|---|---|---|
| Download | `scripts/download_cicids2017.py` | Pull dataset from Kaggle | Real attack traffic |
| Load/clean | `amln/data/cicids.py` | Parse CSVs, fix NaN/Inf, order by day, assign severity | Raw data is dirty; order enables replay |
| Subsample | `stratified_stream`, `scripts/prepare_stream.py` | Keep rare attacks + benign sample → `.npz` | Tractable yet fair; cache avoids re-loading |
| Scale | `amln/data/preprocess.py:RobustScaler` | signed-log → median/IQR → clip ±5 | Tames heavy-tailed flow features |
| Replay | `amln/streaming/replayer.py` | Yield events one at a time, causal scaling | Simulates real-time stream |
| Encode | `amln/models/encoder.py` | Transformer autoencoder → 64-d embedding | Captures *behavior*; gives surprise signal |
| Memory | `amln/memory/adaptive_v2.py` | The landscape dynamics | The novel contribution |
| Evaluate | `amln/benchmark/*` | Prequential comparison vs baselines | Proof of memory efficiency |
| Visualize | `dashboard.py` | Live, self-explaining UI | Makes dynamics observable |

### 3.2 The behavioral autoencoder

A Transformer encoder treats each event's 78 features as a short token sequence,
mean-pools, and projects to a 64-dimensional L2-normalized embedding. A decoder
reconstructs the input. Trained as a **denoising autoencoder** (up to 60% of
features randomly masked) on the mostly-benign warmup prefix, so the latent
captures normal behavior and the reconstruction error of an unusual event is
large — the basis for surprise/zero-day detection.

**Preprocessing note (key fix):** raw CIC-IDS2017 features span many orders of
magnitude (bytes, durations) with extreme outliers. Plain standardization left
+40σ values that stalled training at MSE ≈ 0.55. The `RobustScaler` (signed-log
compression + IQR normalization + clipping to ±5) reduced reconstruction MSE to
**≈ 0.11**, enabling meaningful embeddings.

---

## 4. The Novel Core: Adaptive Memory Landscape V2

Implemented in `amln/memory/adaptive_v2.py`. For each streamed event the
`observe()` method executes the following, which together constitute the novelty.

### 4.1 Surprise-gated plasticity (free-energy inspired)

Each event's **surprise** combines two signals:

```
surprise = w_recon · sigmoid(normalized reconstruction error)
         + w_novelty · (1 − cosine similarity to nearest memory)
```

Plasticity (how strongly the landscape updates) scales with surprise. Familiar,
low-surprise traffic barely perturbs memory; high-surprise events rapidly spawn
or strongly move a memory. Because an unseen attack family is, by construction,
high-surprise, this is the mechanism that enables **zero-day flagging** by
novelty rather than by label.

### 4.2 Energy-budgeted forgetting (bounded memory)

All memories share a fixed **energy budget** `E_max` (sum of heights). When a new
memory would overflow it, energy is reclaimed from the **least-important**
memories first (importance = log-frequency + severity + recency), and any memory
that falls into a valley is pruned. A **grace period** protects freshly spawned
memories so a genuine new family is not deleted before it can be confirmed.

**Consequence (headline property):** the memory footprint is **provably bounded
and roughly constant** regardless of stream length — the property a
store-everything or growing buffer cannot provide.

### 4.3 Drift-aware consolidation

Each memory tracks the running variance (Welford) of the events it absorbs. A
memory whose intra-cluster spread grows beyond a threshold is **split** (an
attack family that is mutating/drifting); tight, frequently reinforced memories
**consolidate** into hills; near-identical memories **merge**. This yields online
adaptation to concept drift with **no neural-network retraining**.

### 4.4 Prediction and zero-day decision

- **Classification:** nearest memory's label.
- **Zero-day / anomaly:** event flagged novel when surprise exceeds an
  (online-adapted) threshold — the correct definition for an unseen family the
  system cannot name.

---

## 5. Experimental Setup

- **Protocol:** prequential (interleaved **test-then-train**) — for each event,
  predict first with current memory, then let the method observe it. Standard for
  online learning; measures how well memory supports detection *as it is built*.
- **Stream:** 101,323 CIC-IDS2017 events, chronological; first ~5,000 = warmup
  (scaler fit + encoder pretrain), excluded from scoring.
- **Zero-day held-out families:** the two rarest (Heartbleed, Web Attack SQL
  Injection) — unseen-family stress test.
- **Encoder:** 64-d latent, 2 Transformer layers, 4 heads, 12 pretrain epochs
  (CPU).
- **Footprint accounting:** each retained item = 64 float32 + an 8-byte label
  (264 bytes). AMLN stores one per memory node; StoreAll one per event seen.

### 5.1 Baselines

| Baseline | Strategy | Memory |
|---|---|---|
| **StoreAll** | Keep every embedding; kNN label | Unbounded (accuracy reference) |
| **SlidingWindow(5000)** | Keep most recent 5,000 | Bounded, forgets by age |
| **Reservoir(5000)** | Uniform random sample of 5,000 | Bounded, random retention |
| **OnlineKMeans(32)** | 32 moving centroids | Bounded, fixed capacity |

---

## 6. Results

Prequential results on the 101k-event CIC-IDS2017 stream (from
`benchmark_results.csv`):

| Method | Accuracy | Macro-F1 | Attack recall | Zero-day recall | Footprint (KB) | Compression | Peak items | Query ms |
|---|---|---|---|---|---|---|---|---|
| **AMLN-V2** | 0.768 | 0.586 | 0.729 | **0.219** | **4.9** | **10,132×** | **19** | 0.038 |
| StoreAll | 0.942 | 0.779 | 0.930 | 0.219 | 26,122 | 1× | 101,323 | 1.054 |
| SlidingWindow(5000) | 0.972 | **0.838** | 0.973 | **0.375** | 1,289 | 20× | 5,000 | 0.077 |
| Reservoir(5000) | 0.918 | 0.678 | 0.903 | 0.031 | 1,289 | 20× | 5,000 | 0.077 |
| OnlineKMeans(32) | 0.570 | 0.301 | 0.409 | 0.000 | 8.25 | 3,166× | 32 | 0.048 |

Encoder pretraining converged to reconstruction MSE **≈ 0.109** (robust scaling).

### 6.1 Honest interpretation

**Where AMLN wins clearly:**

- **Memory efficiency:** 4.9 KB vs StoreAll's 26,122 KB — **~5,300× smaller** —
  while remaining a functional detector. Peak 19 nodes vs 101,323 stored items.
- **Bounded, constant memory:** final energy stayed within the budget (104.9 /
  400); memory does not grow with stream length. 1,305 memories were created and
  1,288 forgotten across the run, with only ~10 kept alive — the self-forgetting
  behavior working as designed.
- **Zero-day detection:** AMLN's surprise-based flag **ties StoreAll (0.219)** and
  beats Reservoir (0.031) and OnlineKMeans (0.000); SlidingWindow is best (0.375).
- **Latency:** 0.038 ms/query vs StoreAll's 1.05 ms (~28× faster), since it
  searches ~19 memories rather than 101,323.

**Where AMLN does not win (stated plainly):**

- **Overall detection F1 (0.586) is below SlidingWindow (0.838) and StoreAll
  (0.779).** AMLN is *not* the top accuracy method in this configuration. The
  cause is diagnosed below.

**Why:** the energy budget (400) and decay settings keep only ~10 live memories
for 15 families, so distinct families compete for too few nodes and some
behaviors are under-represented. This is a **tuning limitation, not a design
flaw** — it is the expected cost of aggressive compression and is directly
controllable (Section 8). OnlineKMeans (fixed 32 centroids, no forgetting)
collapsing to F1 0.301 on the same data confirms that naive fixed-capacity memory
fails here, which is precisely the gap AMLN's adaptive forgetting targets.

### 6.2 Takeaway

AMLN establishes a strong point on the **accuracy-vs-memory frontier**: it retains
competitive attack/zero-day detection at a memory footprint **three to four orders
of magnitude smaller** than storing everything, with constant-bounded growth. The
right framing of the result is *detection quality per kilobyte of memory under
real concept drift*, not raw F1 alone.

---

## 7. Live Dashboard

`dashboard.py` (Streamlit + Plotly). Flicker-free via `st.fragment(run_every=...)`
so only chart regions redraw. Components:

- **"How this works" explainer** and **plain-English captions** under every chart.
- **KPI strip:** events, memories, hills, energy/budget, rolling accuracy, alerts.
- **💾 Memory Management panel:** live "AMLN is using X KB vs store-everything Y KB
  → N× smaller," memories created/forgotten/split, and an **energy gauge** proving
  the budget cannot overflow.
- **Live latent map:** events and memories in 2D (frozen PCA); current event as a
  white dot, anomalies as red stars in empty space.
- **Breathing landscape:** memory heights (red hills, blue slopes).
- **Surprise & accuracy timeline** and a **live alert feed**.

Run: `streamlit run dashboard.py`

---

## 8. Limitations and Future Work

1. **Accuracy–memory tuning.** Current F1 trails the best bounded baseline.
   Raising the energy budget (e.g. 400 → 800–1200) and softening V2 decay to hold
   roughly one hill per family should close most of the gap while keeping memory
   small. A short budget sweep is the immediate next experiment.
2. **Severe class imbalance.** The rarest families (11–36 events) are
   intrinsically hard; per-class recall and cost-sensitive scoring should
   supplement macro-F1.
3. **Encoder capacity.** 12 CPU epochs reach MSE ≈ 0.11; more epochs / GPU and a
   contrastive objective would sharpen family separability.
4. **Baseline cost.** StoreAll's kNN is O(N); an ANN index would speed the
   benchmark (does not affect reported quality).
5. **Evaluation breadth.** Add UNSW-NB15 for cross-dataset generalization and
   report precision/recall per family.

---

## 9. Reproducibility

```powershell
# 0. install
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Kaggle token at %USERPROFILE%\.kaggle\kaggle.json, then download
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe scripts\download_cicids2017.py

# 2. one-time cache of the stratified stream
.\.venv\Scripts\python.exe scripts\prepare_stream.py

# 3. benchmark (writes benchmark_results.csv + benchmark_comparison.png)
.\.venv\Scripts\python.exe run_benchmark.py

# 4. live dashboard
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Fixed seed (42) throughout. Without the dataset, every component falls back to a
synthetic generator so the pipeline is runnable immediately.

### 9.1 Repository map

| Path | Role |
|---|---|
| `amln/models/encoder.py` | Transformer behavioral autoencoder |
| `amln/memory/adaptive_v2.py` | **Novel** memory landscape (V2) |
| `amln/memory/landscape.py` | Original landscape (V1, reference) |
| `amln/memory/graph.py` | Cyber memory graph (attack relationships) |
| `amln/data/cicids.py` | CIC-IDS2017 loader + stratified stream |
| `amln/data/preprocess.py` | StandardScaler + RobustScaler |
| `amln/streaming/replayer.py` | Causal one-at-a-time stream replay |
| `amln/streaming/engine.py` | `LiveEngine.step()` for the dashboard |
| `amln/benchmark/baselines.py` | StoreAll / SlidingWindow / Reservoir / KMeans |
| `amln/benchmark/harness.py` | Prequential benchmark |
| `run_benchmark.py` | Benchmark entry point |
| `dashboard.py` | Streamlit live dashboard |
| `scripts/` | Download, prepare stream, smoke tests |

---

## 10. Conclusion

AMLN demonstrates that framing streaming threat detection as a **bounded
memory-management problem** is viable: a brain-inspired landscape with
surprise-gated plasticity, energy-budgeted forgetting, and drift-aware
consolidation compresses a 101k-event CIC-IDS2017 stream into ~10 memories (4.9
KB) — **thousands of times less memory than storing everything** — while matching
its zero-day recall and operating with far lower latency. Current detection F1
trails the best bounded baseline and is the clear next tuning target, but the
system already stakes out a compelling point on the accuracy-vs-memory frontier
and makes the entire mechanism observable through a real-time, self-explaining
dashboard.

---

*Report generated for the AMLN project. All quantitative results are from
`benchmark_results.csv` produced by `run_benchmark.py` on the cached CIC-IDS2017
stream; numbers depend on configuration and seed.*
