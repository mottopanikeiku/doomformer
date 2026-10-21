"""The examples/ scripts as library code: the autonomous turret of examples/07 really kills."""
import importlib.util
from pathlib import Path

from doomformer import sense
from doomformer.agent import Actuator, Agent


def test_autonomous_turret_kills():
    path = Path(__file__).parents[2] / 'examples' / '07_autonomous_turret.py'
    spec = importlib.util.spec_from_file_location('turret_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    with Agent(example.turret(), sensors=[sense.bearing()],
               actuators={0: Actuator.turn(640), 1: Actuator.turn(-640), 2: Actuator.fire()},
               world=example.arena, world_tics=4) as agent:
        rows = agent.run(32)
        assert all(r['exact'] for r in rows)
        assert rows[-1]['kills'] >= 1
        assert agent.audit()['non_readout_commands'] == 0
        assert agent.calibration['margin'] >= 64
