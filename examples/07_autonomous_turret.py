"""A hand-written integer MLP turns toward and shoots one-hit demons.

The room, one-hit health and freeze/awake schedule are level design, not learned.
Run from the repository root: .venv/bin/python examples/07_autonomous_turret.py
"""
import numpy as np

from doomformer.kit import MLP
from doomformer.agent import Agent, Actuator
from doomformer import sense


def arena(m):
    m.rect(5120, -7168, 7168, -5120, floor=0, ceil=128, tag=20)
    m.thing(6544, -6144, 3002, id=901)
    m.thing(6144, -6144, 1, angle=45)
    return '''
script 220 OPEN {
  Delay(1);
  SetActorProperty(901, APROP_Health, 1);
}
'''


def turret():
    # Staircase activation reconstructs the bearing bin exactly (0..17).
    # Aligned bin 8 gives [left=0,right=0,fire=1].
    return MLP([np.array([[1], [-1], [0]])], [np.array([-8, 8, 1])],
               [np.arange(1, 18)], signed_input=False)


def main():
    with Agent(turret(), sensors=[sense.bearing()],
               actuators={0: Actuator.turn(640), 1: Actuator.turn(-640), 2: Actuator.fire()},
               world=arena, world_tics=4) as agent:
        records = agent.run(32)
        audit = agent.audit()
        kills = records[-1]['kills']
        # agent.run raises df.MismatchError on the first step that differs from the references, so reaching this
        # line means every step was checked and matched; the count below is steps run, not a separate tally.
        print(f"{len(records)} steps, every one checked exact (a mismatch raises); "
              f"kills={kills}; non-readout commands={audit['non_readout_commands']}; "
              f"period={agent.period}; calibration={agent.calibration}")
        if kills < 1 or audit['non_readout_commands'] or not all(r['exact'] for r in records):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
