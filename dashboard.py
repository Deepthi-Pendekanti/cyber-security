"""AMLN live threat-detection dashboard (Streamlit).

Real-time, flicker-free visualization of the adaptive memory landscape as a
CIC-IDS2017 stream is replayed through the encoder + V2 landscape.

Key UI design:
  * A single @st.fragment redraws on a timer (run_every) and updates ONLY the
    chart placeholders in place -- no full-page rerun, so nothing blinks.
  * Live latent map: incoming events (with a fading trail) and memory nodes
    projected into 2D via a frozen PCA, so you literally watch events land near
    the memories they match and anomalies appear in empty space.
  * Breathing landscape: memory heights with hill/valley guides.
  * Surprise timeline + rolling accuracy, and a styled alert feed.

Uses CIC-IDS2017 (cached data/stream.npz) if present, else synthetic.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from amln.config import AMLNConfig
from amln.models.encoder import BehaviorAutoencoder
from amln.streaming import StreamReplayer
from amln.streaming.engine import LiveEngine
from amln.training import pretrain_autoencoder

CICIDS_DIR = Path("data/raw/cicids2017")
STREAM_NPZ = Path("data/stream.npz")

st.set_page_config(page_title="AMLN Live Threat Detection", layout="wide",
                   page_icon="🛡️")

# ---- A little CSS polish --------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 0rem;}
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg,#161a26,#11141d);
        border: 1px solid #2a3042; border-radius: 14px; padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    h1 {letter-spacing:.5px;}
    .alert-crit {color:#ff4b6e; font-weight:600;}
    .alert-ok {color:#2fd47a;}
</style>
""", unsafe_allow_html=True)

# Consistent per-family colors.
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#ff9da6",
           "#9d755d", "#bab0ac", "#eeca3b", "#b279a2", "#ff4b6e", "#17becf",
           "#8c6d31", "#e377c2", "#7f7f7f", "#2ca02c"]


def color_for(name: str) -> str:
    return PALETTE[hash(name) % len(PALETTE)]


# --------------------------------------------------------------------------
# Model + stream (cached: built once per session).
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_parts(max_events: int, epochs: int):
    cfg = AMLNConfig()
    cfg.train.epochs = epochs
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    if STREAM_NPZ.exists():
        z = np.load(STREAM_NPZ, allow_pickle=True)
        cfg.encoder.num_features = z["features"].shape[1]
        feats = z["features"][:max_events]
        labels = z["labels"][:max_events]
        severity = z["severity"][:max_events]
        ts = z["timestamps"][:max_events]
        names = {int(i): str(n) for i, n in zip(z["label_ids"], z["label_names"])}
        source = "CIC-IDS2017 (real traffic)"
    else:
        from amln.data import ATTACK_FAMILIES, CyberEventGenerator
        gen = CyberEventGenerator(num_features=cfg.encoder.num_features,
                                  seed=cfg.train.seed)
        batch = gen.sample(max_events)
        feats, labels, severity = batch.features, batch.labels, batch.severity
        ts = np.arange(len(batch), dtype=np.float64)
        names = {i: n for i, n in enumerate(ATTACK_FAMILIES)}
        source = "Synthetic (CIC-IDS2017 not downloaded)"

    warmup = min(5000, len(feats) // 3)
    scaler = "robust" if STREAM_NPZ.exists() else "standard"
    replayer = StreamReplayer(feats, labels, severity, ts, warmup=warmup,
                              scaler=scaler)
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, replayer.standardized_block(0, warmup), cfg)
    return model, replayer, cfg, names, source


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️  Controls")
    max_events = st.slider("Stream size", 3000, 101000, 30000, step=1000)
    epochs = st.slider("Encoder epochs", 2, 20, 8)
    speed = st.slider("Events / tick", 5, 300, 60)
    refresh = st.select_slider("Refresh rate", options=[0.25, 0.5, 1.0, 2.0],
                               value=0.5, format_func=lambda s: f"{s:g}s")
    st.divider()
    c1, c2 = st.columns(2)
    go_btn = c1.button("▶ Start", width="stretch", type="primary")
    pause_btn = c2.button("⏸ Pause", width="stretch")
    reset_btn = st.button("↺ Reset", width="stretch")

with st.spinner("Building encoder + stream (one-time) ..."):
    model, replayer, cfg, names, source = build_parts(max_events, epochs)
st.sidebar.success(f"Data: {source}")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
ss = st.session_state
if "engine" not in ss or reset_btn:
    ss.engine = LiveEngine(model, replayer, cfg, names)
    ss.running = False
    ss.surprise_hist = deque(maxlen=250)
    ss.acc_hist = deque(maxlen=250)
    ss.node_hist = deque(maxlen=250)
    ss.trail = deque(maxlen=60)          # recent event positions (fading)
    ss.log = deque(maxlen=12)
    ss.alerts_feed = deque(maxlen=8)
    ss.last = None
    ss.done = False
if go_btn:
    ss.running = True
if pause_btn:
    ss.running = False

engine: LiveEngine = ss.engine

st.title("🛡️  AMLN — Adaptive Memory Landscape · Live Threat Detection")

with st.expander("❓ How this works — read me first (30-second version)", expanded=False):
    st.markdown("""
**The idea in one line:** a security system that watches network traffic
**one event at a time** and remembers attack patterns using a tiny, *living*
memory — instead of storing every event forever.

**What the simulation does:** we replay **real recorded network traffic**
(CIC-IDS2017 — a real week of a network with real attacks mixed in) as if it
were happening live. For each event the system asks:
*"Have I seen behavior like this before, or is it new?"*

- **Seen before →** it strengthens the matching **memory** (a "hill").
- **New →** it creates a fresh memory.
- **Not seen in a while →** that memory **fades and is forgotten**.
- **Very unfamiliar →** it raises an **🚨 alert** (possible zero-day attack).

**Why it's special (the novelty):** the memory has a **fixed size budget**, so it
can *never* grow out of control. It keeps matching accuracy as high as a system
that stores *everything*, while using **thousands of times less memory**. The
"💾 Memory Management" panel below shows that saving live.

**To see the action:** press **▶ Start** and let the *Events* counter climb into
the thousands. Early on it's all `BENIGN` (Monday = normal traffic); attacks and
red-star anomalies appear as the week's stream progresses.
    """)

status_slot = st.empty()
kpi_slot = st.empty()
mem_slot = st.empty()
row1 = st.container()
row2 = st.container()
with row1:
    cL, cR = st.columns([3, 2], gap="medium")
    map_slot = cL.empty()
    land_slot = cR.empty()
with row2:
    cA, cB = st.columns([3, 2], gap="medium")
    surp_slot = cA.empty()
    feed_slot = cB.empty()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def draw(snap):
    # KPI cards
    with kpi_slot.container():
        k = st.columns(6)
        k[0].metric("Events", f"{engine.processed:,}")
        k[1].metric("Memories", snap.num_nodes if snap else 0,
                    help="Active memory nodes (bounded by energy budget)")
        k[2].metric("Hills", snap.hills if snap else 0)
        k[3].metric("Energy", f"{snap.total_energy:.0f}/{snap.energy_budget:.0f}"
                    if snap else "—")
        k[4].metric("Rolling acc", f"{snap.rolling_accuracy:.1%}" if snap else "—")
        k[5].metric("🚨 Alerts", f"{engine.alerts:,}")

    if snap is None:
        return

    # ---- 💾 MEMORY MANAGEMENT panel (the headline novelty) ------------
    with mem_slot.container():
        st.markdown("### 💾 Memory Management — the whole point")
        m = st.columns([1, 1, 1, 2])
        m[0].metric("AMLN is using", f"{snap.amln_kb:.1f} KB",
                    help="Total memory AMLN actually keeps right now.")
        m[1].metric("Store-everything needs", f"{snap.storeall_kb:,.0f} KB",
                    help="What a system that saved every event would use by now.")
        m[2].metric("➡️ Memory saved", f"{snap.savings_x:,.0f}× smaller",
                    delta=f"{snap.total_pruned:,} forgotten",
                    help="How many times less memory AMLN uses.")
        # Energy gauge: proof the memory is bounded.
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=snap.total_energy,
            number={"suffix": f" / {snap.energy_budget:.0f}", "font": {"size": 22}},
            title={"text": "Memory energy used (capped — can't overflow)",
                   "font": {"size": 13}},
            gauge={"axis": {"range": [0, snap.energy_budget]},
                   "bar": {"color": "#ff4b6e"},
                   "steps": [{"range": [0, snap.energy_budget * 0.7],
                              "color": "#1e2430"},
                             {"range": [snap.energy_budget * 0.7,
                                        snap.energy_budget], "color": "#3a2230"}]}))
        gauge.update_layout(template="plotly_dark", height=160,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        m[3].plotly_chart(gauge, width="stretch",
                          key=f"gauge_{engine.processed}")
        st.caption(
            f"🧠 **{snap.total_spawned:,} memories created · "
            f"{snap.total_pruned:,} forgotten · {snap.total_splits:,} split** "
            f"→ only **{snap.num_nodes} kept alive** right now. "
            f"AMLN compressed **{engine.processed:,} events** into these few "
            f"memories. A store-everything system would already hold all "
            f"{engine.processed:,} of them. That gap is the saving, and because "
            f"of the energy cap it stays small forever — the memory never "
            f"blows up no matter how long the stream runs.")
        st.divider()

    # ---- Live latent map (events + memories in 2D) --------------------
    fig = go.Figure()
    # fading event trail
    if ss.trail:
        tx = [p[0] for p in ss.trail]
        ty = [p[1] for p in ss.trail]
        op = np.linspace(0.1, 0.5, len(tx))
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="markers",
                                 marker=dict(size=6, color="#4c78a8",
                                             opacity=list(op)),
                                 hoverinfo="skip", showlegend=False))
    # memory nodes sized by count, colored by family
    if snap.node_xy:
        nx = [p[0] for p in snap.node_xy]
        ny = [p[1] for p in snap.node_xy]
        sizes = 14 + 30 * (np.array(snap.node_heights) /
                           (max(snap.node_heights) + 1e-9))
        ncol = [color_for(l) for l in snap.node_labels]
        fig.add_trace(go.Scatter(
            x=nx, y=ny, mode="markers+text",
            marker=dict(size=sizes, color=ncol, line=dict(width=2, color="#fff"),
                        opacity=0.9),
            text=snap.node_labels, textposition="top center",
            textfont=dict(size=9, color="#cfd6e4"),
            hovertext=[f"{l}<br>height {h:.1f} · seen {c}"
                       for l, h, c in zip(snap.node_labels, snap.node_heights,
                                          snap.node_counts)],
            hoverinfo="text", showlegend=False))
    # current event: red star if anomaly else white dot
    ex, ey = snap.event_xy
    fig.add_trace(go.Scatter(
        x=[ex], y=[ey], mode="markers",
        marker=dict(size=20 if snap.is_anomaly else 13,
                    color="#ff4b6e" if snap.is_anomaly else "#ffffff",
                    symbol="star" if snap.is_anomaly else "circle",
                    line=dict(width=2, color="#ff4b6e" if snap.is_anomaly
                              else "#4c78a8")),
        hovertext=[f"event #{snap.index}<br>{snap.true_label_name}"],
        hoverinfo="text", showlegend=False))
    fig.update_layout(
        title="🧠 Memory landscape — live latent map",
        template="plotly_dark", height=430, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    map_slot.plotly_chart(fig, width="stretch",
                          key=f"map_{engine.processed}")
    cL.caption(
        "🗺️ **What am I looking at?** Every behavior becomes a point in "
        "'behavior space' — similar traffic sits close together. **Orange dots = "
        "memories** (bigger = stronger). The **white dot = the event happening "
        "now**; watch it land next to the memory it matches. A **red ⭐ in empty "
        "space = an anomaly** the system has no memory for (possible zero-day).")

    # ---- Breathing landscape (heights) --------------------------------
    order = np.argsort(snap.node_heights)[::-1]
    heights = [snap.node_heights[i] for i in order]
    labels = [snap.node_labels[i] for i in order]
    colors = ["#ff4b6e" if h >= cfg.landscape.hill_threshold else "#4c78a8"
              for h in heights]
    f2 = go.Figure(go.Bar(x=labels, y=heights, marker_color=colors,
                          marker_line_width=0))
    f2.add_hline(y=cfg.landscape.hill_threshold, line_dash="dash",
                 line_color="#ff4b6e", annotation_text="hill")
    f2.update_layout(title="⛰️ Memory heights (red = hill)",
                     template="plotly_dark", height=430,
                     margin=dict(l=10, r=10, t=40, b=60),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     yaxis_title="stability")
    f2.update_xaxes(tickangle=-40)
    land_slot.plotly_chart(f2, width="stretch",
                           key=f"land_{engine.processed}")
    cR.caption(
        "⛰️ **Memory strength.** Each bar is one memory; taller = more "
        "established. **Red bars are 'hills'** (strong long-term memories seen "
        "many times); **blue bars are 'slopes'** (new, still forming). Bars grow "
        "when a pattern repeats and shrink/vanish when it's not seen — the memory "
        "literally breathes as traffic flows.")

    # ---- Surprise + accuracy timeline ---------------------------------
    f3 = go.Figure()
    f3.add_trace(go.Scatter(y=list(ss.surprise_hist), mode="lines",
                            name="surprise", line=dict(color="#f58518", width=2),
                            fill="tozeroy", fillcolor="rgba(245,133,24,0.15)"))
    f3.add_trace(go.Scatter(y=list(ss.acc_hist), mode="lines",
                            name="rolling acc", line=dict(color="#2fd47a", width=2),
                            yaxis="y2"))
    f3.add_hline(y=engine.landscape.cfg2.anomaly_surprise, line_dash="dot",
                 line_color="#ff4b6e")
    f3.update_layout(title="📈 Surprise & detection accuracy",
                     template="plotly_dark", height=330,
                     margin=dict(l=10, r=10, t=40, b=10),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     yaxis=dict(title="surprise", range=[0, 1]),
                     yaxis2=dict(title="acc", overlaying="y", side="right",
                                 range=[0, 1]),
                     legend=dict(orientation="h", y=1.15))
    surp_slot.plotly_chart(f3, width="stretch",
                           key=f"surp_{engine.processed}")
    cA.caption(
        "📈 **Orange = surprise** (how unfamiliar each event is). It stays low "
        "for normal traffic and **spikes when an attack arrives**. The dotted red "
        "line is the alert threshold — cross it and the event is flagged. "
        "**Green = accuracy** (how often the system names the event type correctly).")

    # ---- Alert feed ----------------------------------------------------
    lines = ["#### 🚨 Live alert feed"]
    if not ss.alerts_feed:
        lines.append("_No anomalies yet — monitoring..._")
    for a in reversed(ss.alerts_feed):
        lines.append(f"<span class='alert-crit'>▲ #{a['i']} · {a['name']} "
                     f"· surprise {a['s']:.2f} · sev {a['sev']:.2f}</span>  ")
    lines.append("\n#### 📋 Recent events")
    for e in reversed(ss.log):
        icon = "🟢" if e["correct"] else "🔴"
        lines.append(f"{icon} `#{e['i']:>6}` **{e['true']}** → {e['pred']} "
                     f"(surp {e['surp']:.2f})  ")
    feed_slot.markdown("\n".join(lines), unsafe_allow_html=True)
    cB.caption(
        "🚨 **Live feed.** Top: events flagged as anomalies (novel/suspicious). "
        "Bottom: the most recent events with 🟢 = predicted correctly, "
        "🔴 = mispredicted, and the surprise score for each.")


# --------------------------------------------------------------------------
# The live loop as a FRAGMENT: reruns itself on a timer, updating only this
# fragment's placeholders -> smooth, no full-page blink.
# --------------------------------------------------------------------------
@st.fragment(run_every=refresh if ss.get("running") else None)
def live_tick():
    if not ss.running or ss.done:
        if ss.last is not None:
            draw(ss.last)
        return
    snap = None
    for _ in range(speed):
        snap = engine.step()
        if snap is None:
            ss.done = True
            ss.running = False
            break
        ss.surprise_hist.append(snap.surprise)
        ss.acc_hist.append(snap.rolling_accuracy)
        ss.node_hist.append(snap.num_nodes)
        ss.trail.append(snap.event_xy)
        ss.log.append({"i": snap.index, "true": snap.true_label_name,
                       "pred": snap.pred_label_name, "surp": snap.surprise,
                       "correct": snap.correct})
        if snap.is_anomaly:
            ss.alerts_feed.append({"i": snap.index, "name": snap.true_label_name,
                                   "s": snap.surprise, "sev": snap.severity})
    ss.last = snap or ss.last
    draw(ss.last)


# Status line
if ss.done:
    status_slot.success(f"✅ Stream complete — {engine.processed:,} events, "
                        f"{engine.alerts:,} alerts, {len(engine.landscape)} memories.")
elif ss.running:
    status_slot.info("🟢 Streaming live...")
elif engine.processed == 0:
    status_slot.info("Press ▶ Start in the sidebar to begin the live stream.")
else:
    status_slot.warning("⏸ Paused.")

live_tick()
