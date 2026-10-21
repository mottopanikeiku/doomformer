"""Copy this complete two-sensor agent and replace the world, weights and wiring.

All runtime inputs and action selection originate in ACS; Python verifies only.
"""
import numpy as np

from doomformer.agent import Agent, Actuator
from doomformer.kit import MLP
from doomformer import sense


def your_world(m):
    m.rect(5120, -7168, 7168, -5120, floor=0, ceil=128, tag=20)
    m.thing(6544, -6144, 3002, id=901)  # all sensed monsters share this TID
    m.thing(6144, -6144, 1, angle=0)   # last player start is the real marine
    return '''script 220 OPEN { Delay(1); SetActorProperty(901, APROP_Health, 1); }'''


def main():
    # Inputs: bearing bin, live-monster count. Three raw output action scores.
    net = MLP([np.array([[1, 0], [-1, 0], [0, 2]])],
              [np.array([-8, 8, -1])], [np.arange(1, 18)], signed_input=False)
    with Agent(net, sensors=[sense.bearing(), sense.monster_count()],
               actuators={'y0': Actuator.turn(640), 'y1': Actuator.turn(-640),
                          'y2': Actuator.fire()}, world=your_world,
               monster_tid=901, world_tics=4) as agent:
        for row in agent.run(8):
            print(f"step={row['step']} inputs={row['inputs']} "
                  f"action={row['action']} exact={row['exact']} kills={row['kills']}")
        print(agent.audit())
        if agent.audit()['non_readout_commands']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
