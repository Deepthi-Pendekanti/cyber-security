"""Brain-Inspired Adaptive Memory Landscape Network (AMLN).

A biologically inspired continual-learning framework for cyber threat
detection. The novelty lives in :mod:`amln.memory.landscape`, which models
long-term memory formation (hills), short-term memory (slopes), and
intelligent forgetting (decay into valleys) on top of a learned 64-d latent
space produced by :mod:`amln.models.encoder`.

Key components:
  - :mod:`amln.memory.landscape` - Adaptive Memory Landscape (hills/slopes/valleys)
  - :mod:`amln.memory.graph` - Cyber Memory Graph (attack relationships)
  - :mod:`amln.memory.reconstruction` - Memory Reconstruction Engine
  - :mod:`amln.memory.prediction` - Threat Evolution Prediction Engine
  - :mod:`amln.training.feedback` - Closed-loop Adaptive Learning
  - :mod:`amln.visualization` - Visualization utilities
"""

from amln.config import AMLNConfig

__all__ = ["AMLNConfig"]
__version__ = "0.2.0"
