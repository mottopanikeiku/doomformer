"""DOOMFORMER: integer neural networks whose arithmetic is executed by DOOM game mechanics (ViZDoom).

    import doomformer as df
    net = df.MLP(weights, biases, thresholds)       # or df.MLP.from_torch(model, calib_x)
    with df.compile(net) as brain:
        y = brain(x)                                 # computed by DOOM, checked against net.reference(x)

    df.Transformer.load("doomnames_onemap")          # a shipped trained model, by name (df.available_models())
    df.with_argmax(net)                              # + winner layers: DOOM outputs the one-hot of the argmax
    df.run_many(net, xs, workers=8)                  # many passes over many DOOM games (backend="ray": a cluster)

See doomformer.kit for what DOOM computes and what Python does.
"""
from . import act, record, sense
from .agent import Actuator, Agent
from .drawpad import with_argmax
from .fleet import run_many
from .interop import Staircase, staircase
from .kit import MLP, Brain, MismatchError, RangeError, SpikingNet, Transformer, compile, plan, play
from .learnkit import Learner, Perceptron
from .record import Recording
from .shipped import names as available_models

__all__ = ["MLP", "Brain", "MismatchError", "RangeError", "SpikingNet", "Transformer", "act", "compile", "plan",
           "play", "sense", "record", "Agent", "Actuator", "staircase", "Staircase", "Perceptron", "Learner",
           "Recording", "available_models", "with_argmax", "run_many"]
