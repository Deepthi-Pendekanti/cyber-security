# AMLN — Adaptive Memory Landscape Network

A brain-inspired, **bounded-memory** framework for **real-time streaming cyber
threat detection**. Instead of storing every event (or a fixed window of them),
AMLN compresses a continuous log stream into a small, self-organizing *memory
landscape* of hills (long-term patterns), slopes (short-term memory), and
valleys (forgotten), updating online with **no network retraining**.

The headline claim, measured by the benchmark harness:

> AMLN matches a store-everything detector's accuracy while using **orders of
> magnitude less memory** (bounded, roughly constant regardless of stream
> length) and with far lower query latency — benchmarked on CIC-IDS2017
> replayed as a live stream.

---

## What's novel

The core research contribution lives in `amln/memory/adaptive_v2.py`
(`AdaptiveMemoryLandscapeV2`), which adds three mechanisms on top of the
hills/slopes/valleys terrain:

1. **Surprise-gated plasticity** (free-energy inspired). Each event's *surprise*
   combines encoder reconstruction error with novelty (distance to the nearest
   memory). High-surprise events form memories fast; familiar events barely
   perturb the landscape. This is what gives sensitive **zero-day** detection.
2. **Energy-budgeted forgetting**. The landscape shares a fixed energy budget;
   when a new memory would overflow it, energy is reclaimed from the least
   important memories first. This makes the memory footprint **provably bounded
   and roughly constant** — the headline benchmark property. A grace period
   protects freshly spawned memories so a genuine new family isn't forgotten
   before it can be confirmed.
3. **Drift-aware consolidation**. Each memory tracks the running variance of the
   events it absorbs; a drifting cluster (a mutating attack family) is **split**,
   while tight, frequently reinforced clusters consolidate into hills.

## Architecture

```
Log stream (CIC-IDS2017 replayed, or synthetic)
        │  one event at a time, in timestamp order
        ▼
Transformer behavioral autoencoder  (amln/models/encoder.py)
        │  64-d embedding + reconstruction error (surprise signal)
        ▼
Adaptive Memory Landscape V2  (amln/memory/adaptive_v2.py)
   observe() → spawn / reinforce / split / decay / prune (energy-bounded)
        │
        ├──────────────► Live dashboard   (dashboard.py)
        └──────────────► Benchmark harness (run_benchmark.py)
```

## Project layout

| Path | Purpose |
|------|---------|
| `amln/models/encoder.py` | Transformer autoencoder → 64-d behavioral embedding |
| `amln/memory/adaptive_v2.py` | **Novel** surprise-gated, energy-bounded, drift-aware landscape |
| `amln/memory/landscape.py` | Original V1 landscape (kept for comparison) |
| `amln/streaming/replayer.py` | Causal, timestamp-ordered stream replay |
| `amln/streaming/engine.py` | `LiveEngine.step()` — one event → snapshot (drives the UI) |
| `amln/benchmark/baselines.py` | StoreAll, SlidingWindow, Reservoir, OnlineKMeans |
| `amln/benchmark/harness.py` | Prequential (test-then-train) benchmark |
| `amln/data/cicids.py` | CIC-IDS2017 loader (chronological, cleaned) |
| `dashboard.py` | Streamlit live threat-detection dashboard |
| `run_benchmark.py` | Runs the memory-management benchmark, exports CSV + plot |
| `scripts/download_cicids2017.py` | Downloads the dataset from Kaggle |

## Setup

```powershell
# From the project root (Windows PowerShell)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Get the dataset (one-time)

CIC-IDS2017 is a recorded, timestamped dataset that AMLN **replays as a live
stream** (there is no truly live public feed; replay-in-order is the standard
and reproducible approach).

1. Create a Kaggle API token: kaggle.com → Settings → API → **Create New Token**.
2. Save the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json`.
3. Download:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe scripts\download_cicids2017.py
```

Files land in `data/raw/cicids2017/`. Both the dashboard and the benchmark
**auto-detect** the dataset and use it; without it they fall back to a synthetic
stream so everything is runnable immediately.

## Run the live dashboard

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open the printed `localhost` URL, set the controls in the sidebar, and press
**Start**. You'll watch memories grow into hills, stale ones collapse into
valleys, a live surprise/anomaly timeline, rolling accuracy, and a scrolling
event log with zero-day alerts.

## Run the benchmark

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe run_benchmark.py
```

Prints a comparison table and writes `benchmark_results.csv` +
`benchmark_comparison.png` (detection quality and accuracy-vs-memory).

### Metrics reported

- **accuracy / macro-F1** (prequential online evaluation)
- **attack recall** and **zero-day recall**
- **memory footprint (KB)** and **compression ratio**
- **peak / final items retained** (the bounded-memory check)
- **update & query latency (ms/event)**

## Legacy demo

The original one-shot pipeline is still available:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe main.py
```

## Notes

- On the synthetic fallback, families are trivially separable, so most methods
  score ~0.99 — this only sanity-checks the pipeline. The **real CIC-IDS2017
  stream** (overlapping flows, class imbalance, week-long drift) is where
  AMLN's bounded-memory adaptivity separates it from the baselines.
- `scripts/smoke_v2.py` and `scripts/smoke_engine.py` are quick, dataset-free
  self-tests of the landscape and the live engine.
