"""End-to-end AMLN demo on synthetic cyber telemetry.

Pipeline:
  1. Generate structured cyber events (some attack families held out as zero-day).
  2. Pre-train the Transformer autoencoder so the 64-d latent space is meaningful.
  3. Stream training events into the adaptive memory landscape (continual, no
     network retraining) - watch slopes grow into hills and stale memories decay.
  4. Build the Cyber Memory Graph capturing temporal transitions.
  5. Evaluate: classification via nearest-hill retrieval, pattern completion from
     partial events, zero-day detection on unseen families, and memory metrics.
  6. Attack reconstruction from partial evidence (Memory Reconstruction Engine).
  7. Threat evolution prediction (what attack stage comes next?).
  8. Closed-loop adaptive learning feedback demonstration.
  9. Visualization (if matplotlib available).

Run:  python main.py
"""

from __future__ import annotations

import numpy as np
import torch

from amln.config import AMLNConfig
from amln.data import ATTACK_FAMILIES, CyberEventGenerator
from amln.data.preprocess import StandardScaler
from amln.evaluation import (
    memory_footprint,
    reconstruction_accuracy,
    retrieval_accuracy,
    zero_day_detection,
)
from amln.memory.landscape import AdaptiveMemoryLandscape, RegionType
from amln.memory.graph import CyberMemoryGraph
from amln.memory.reconstruction import MemoryReconstructionEngine
from amln.memory.prediction import ThreatEvolutionEngine
from amln.models.encoder import BehaviorAutoencoder
from amln.training import populate_landscape, pretrain_autoencoder
from amln.training.feedback import AdaptiveFeedbackLoop, FeedbackEvent


def _section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    cfg = AMLNConfig()
    if torch.cuda.is_available():
        cfg.train.device = "cuda"
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # Hold out two families as "zero-day" - never seen during landscape build.
    zero_day = [ATTACK_FAMILIES.index("apt_persistence"),
                ATTACK_FAMILIES.index("insider_threat")]
    known = [i for i in range(len(ATTACK_FAMILIES)) if i not in zero_day]

    _section("1) Generating synthetic cyber events")
    gen = CyberEventGenerator(num_features=cfg.encoder.num_features,
                              seed=cfg.train.seed)
    train = gen.sample(12000, families=known)
    test_known = gen.sample(3000, families=known)
    test_zero_day = gen.sample(1500, families=zero_day)
    print(f"  train={len(train)}  test_known={len(test_known)}  "
          f"test_zero_day={len(test_zero_day)}")
    print(f"  known families:    {[ATTACK_FAMILIES[i] for i in known]}")
    print(f"  zero-day families: {[ATTACK_FAMILIES[i] for i in zero_day]}")

    # Standardize on the training split.
    scaler = StandardScaler().fit(train.features)
    x_train = scaler.transform(train.features)
    x_test = scaler.transform(test_known.features)
    x_zero = scaler.transform(test_zero_day.features)

    _section("2) Pre-training the behavioral autoencoder (Transformer encoder)")
    model = BehaviorAutoencoder(cfg.encoder)
    pretrain_autoencoder(model, x_train, cfg)

    _section("3) Continual population of the adaptive memory landscape")
    landscape = AdaptiveMemoryLandscape(cfg.encoder.latent_dim, cfg.landscape)
    graph = CyberMemoryGraph(landscape)
    metrics_log: list[dict] = []
    populate_landscape(model, landscape, x_train, train.labels, train.severity,
                       cfg, log_every=3000, graph=graph, metrics_log=metrics_log)
    reg = landscape.regions()
    print(f"  final: nodes={len(landscape)} hills={reg[RegionType.HILL]} "
          f"slopes={reg[RegionType.SLOPE]}")

    _section("4) Building Cyber Memory Graph (similarity edges)")
    graph.build_from_landscape(similarity_threshold=0.7)
    print(f"  graph edges: {graph.num_edges()}")
    print(f"  mean edge weight: {graph.mean_edge_weight():.3f}")
    hubs = graph.hub_nodes(top_k=5)
    print(f"  top hub nodes (out-degree): {hubs}")

    _section("5a) Classification via nearest-hill retrieval (known families)")
    ret = retrieval_accuracy(model, landscape, x_test, test_known.labels, cfg)
    print(f"  accuracy={ret['accuracy']:.4f}  precision={ret['precision']:.4f}  "
          f"recall={ret['recall']:.4f}  f1={ret['f1']:.4f}")
    print(f"  retrieval_latency={ret['retrieval_latency_ms']:.4f} ms/event")

    _section("5b) Behavior-driven pattern completion (partial events)")
    partial = gen.make_partial(x_test, keep_ratio=0.4)
    rec = reconstruction_accuracy(model, landscape, partial, x_test, cfg)
    print(f"  pattern_reconstruction_cosine={rec['pattern_reconstruction_cosine']:.4f}"
          f"  match_rate={rec['match_rate']:.4f}")

    _section("5c) Zero-day detection (unseen attack families)")
    zd = zero_day_detection(model, landscape, x_zero, test_zero_day.labels, cfg)
    print(f"  zero_day_detection_rate={zd['zero_day_detection_rate']:.4f}  "
          f"(over {zd['evaluated_attacks']} attack events)")

    _section("5d) Memory-specific metrics")
    fp = memory_footprint(landscape, cfg.encoder.latent_dim,
                          cfg.encoder.num_features)
    print(f"  memory_nodes={fp['memory_nodes']}")
    print(f"  compression_ratio={fp['compression_ratio']:.1f}x")
    print(f"  forgetting_efficiency={fp['forgetting_efficiency']:.4f}")
    print(f"  storage_reduction={fp['storage_reduction'] * 100:.2f}%")
    print(f"  raw={fp['raw_kb']:.1f} KB  ->  memory={fp['memory_kb']:.1f} KB")

    _section("6) Attack Reconstruction from partial evidence")
    recon_engine = MemoryReconstructionEngine(model, landscape, graph, cfg)
    # Take a few partial events and reconstruct
    sample_partial = gen.make_partial(x_test[:20], keep_ratio=0.3)
    result = recon_engine.reconstruct(sample_partial)
    print(f"  reconstruction confidence: {result.reconstruction_confidence:.4f}")
    print(f"  coverage (matched): {result.coverage:.4f}")
    if result.attack_chain:
        print(f"  attack chain length: {len(result.attack_chain.node_indices)}")
        print(f"  chain confidence: {result.attack_chain.confidence:.4f}")
        if result.predicted_stages:
            print(f"  predicted next stages: {len(result.predicted_stages)}")

    # Explainability for a single event
    report = recon_engine.explain(sample_partial[0])
    print(f"\n  Attribution Report (sample event):")
    print(f"    {report.explanation}")
    if report.contributing_factors:
        for k, v in report.contributing_factors.items():
            print(f"    {k}: {v:.4f}")

    _section("7) Threat Evolution Prediction")
    prediction_engine = ThreatEvolutionEngine(landscape, graph, cfg)
    # Assess threat for the top few hills
    hill_nodes = [i for i, n in enumerate(landscape.nodes)
                  if n.region(cfg.landscape) == RegionType.HILL]
    if hill_nodes:
        sample_hill = hill_nodes[0]
        threat_report = prediction_engine.assess_threat(sample_hill)
        node = landscape.nodes[sample_hill]
        print(f"  Assessing hill node {sample_hill} "
              f"(label={ATTACK_FAMILIES[node.label]}, "
              f"severity={node.severity:.2f}):")
        print(f"    risk_score={threat_report.risk_score:.4f}")
        print(f"    escalation_probability={threat_report.escalation_probability:.4f}")
        print(f"    max_predicted_severity={threat_report.max_predicted_severity:.4f}")
        print(f"    next stages predicted: {len(threat_report.next_stages)}")
        print(f"    trajectory length: {threat_report.trajectory.trajectory_length}")
        for i, stage in enumerate(threat_report.next_stages[:3]):
            label_name = ATTACK_FAMILIES[stage.node_label] if stage.node_label < len(
                ATTACK_FAMILIES) else f"family_{stage.node_label}"
            print(f"      -> stage {i+1}: {label_name} "
                  f"(prob={stage.probability:.3f}, sev={stage.severity:.2f})")

        # Batch risk assessment
        risks = prediction_engine.batch_risk_assessment()
        print(f"\n  Batch risk assessment ({len(risks)} hills):")
        for idx, risk in risks[:5]:
            n = landscape.nodes[idx]
            label = ATTACK_FAMILIES[n.label] if n.label < len(ATTACK_FAMILIES) else "?"
            print(f"    node {idx} [{label}]: risk={risk:.4f}")

    _section("8) Closed-loop Adaptive Learning (feedback simulation)")
    feedback_loop = AdaptiveFeedbackLoop(landscape, graph, cfg)
    # Simulate feedback: compare predictions vs actual for test set
    predictions = ret["predictions"]
    n_feedback = min(500, len(predictions))
    feedback_events = []
    for i in range(n_feedback):
        node_match, _ = landscape.retrieve(
            model.embed(torch.from_numpy(x_test[i:i+1].astype(np.float32))
                        ).numpy()[0], hills_only=False)
        node_idx = -1
        if node_match:
            for j, n in enumerate(landscape.nodes):
                if n is node_match:
                    node_idx = j
                    break
        feedback_events.append(FeedbackEvent(
            node_idx=node_idx,
            predicted_label=int(predictions[i]),
            actual_label=int(test_known.labels[i]),
            predicted_severity=float(node_match.severity) if node_match else 0.0,
            actual_severity=float(test_known.severity[i]),
        ))

    stats = feedback_loop.process_batch(feedback_events)
    summary = feedback_loop.summary()
    print(f"  feedback processed: {summary['total_feedback']}")
    print(f"  prediction accuracy: {summary['accuracy']:.4f}")
    print(f"  severity MSE: {summary['severity_mse']:.6f}")
    print(f"  reinforcements applied: {summary['reinforcements']}")
    print(f"  decays applied: {summary['decays']}")
    print(f"  nodes with adapted thresholds: {summary['adapted_nodes']}")

    # Show landscape state after feedback
    reg_after = landscape.regions()
    print(f"\n  landscape after feedback: nodes={len(landscape)} "
          f"hills={reg_after[RegionType.HILL]} slopes={reg_after[RegionType.SLOPE]}")

    _section("9) Visualization")
    try:
        from amln.visualization import (
            plot_latent_space,
            plot_memory_heights,
            plot_memory_graph,
            plot_continual_metrics,
        )
        # Embed test data for visualization
        with torch.no_grad():
            test_emb = model.embed(
                torch.from_numpy(x_test[:2000].astype(np.float32))
            ).numpy()

        print("  Generating plots...")
        plot_latent_space(test_emb, test_known.labels[:2000], landscape,
                          method="pca", title="AMLN Latent Space (PCA)",
                          save_path="amln_latent_space.png")
        plot_memory_heights(landscape, save_path="amln_memory_heights.png")
        plot_memory_graph(graph, landscape, save_path="amln_memory_graph.png")
        if metrics_log:
            plot_continual_metrics(metrics_log, save_path="amln_continual_metrics.png")
    except ImportError as e:
        print(f"  (visualization skipped: {e})")
        print("  Install matplotlib for plots: pip install matplotlib")

    print("\n" + "=" * 68)
    print("AMLN Demo Complete")
    print("=" * 68)


if __name__ == "__main__":
    main()
