"""DOOMFORMER: integer neural networks whose arithmetic is executed by DOOM game mechanics (ViZDoom).

    import doomformer as df
    net = df.MLP(weights, biases, thresholds)       # or df.MLP.from_torch(model, calib_x)
    with df.compile(net) as brain:
        y = brain(x)                                 # computed by DOOM, checked against net.reference(x)

    df.Transformer.load("doomnames_onemap")          # a shipped trained model, by name (df.available_models())
    df.with_argmax(net)                              # + winner layers: DOOM outputs the one-hot of the argmax
    df.run_many(net, xs, workers=8)                  # many passes over many DOOM games (backend="ray": a cluster)
    df.Machine(program, registers={...}).run()       # a counter machine in one map (docs/machine.md), every step checked

See doomformer.kit for what DOOM computes and what Python does.
"""
from . import act, intervene, record, sense
from .agent import Actuator, Agent
from .drawpad import with_argmax
from .fleet import run_many
from .interop import Staircase, staircase
from .kit import (MLP, Brain, MismatchError, RangeError, ReadoutError, SpikingNet, Transformer, compile, plan,
                  play)
from .learnkit import Learner, Perceptron
from .machine import Machine
from .record import Recording
from .schedule import check_schedule
from .shipped import names as available_models
from .stored import StoredMachine

__all__ = ["MLP", "Brain", "MismatchError", "RangeError", "ReadoutError", "SpikingNet", "Transformer", "act",
           "compile", "plan",
           "play", "sense", "record", "Agent", "Actuator", "staircase", "Staircase", "Perceptron", "Learner",
           "Recording", "available_models", "with_argmax", "run_many", "intervene", "patch_recovery", "Machine",
           "StoredMachine", "check_schedule"]


def __getattr__(name):                                   # df.patch_recovery; lazy so `python -m doomformer.patching` runs clean
    if name == "patch_recovery":
        from .patching import patch_recovery
        return patch_recovery
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
