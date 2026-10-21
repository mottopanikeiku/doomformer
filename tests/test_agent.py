"""Real DOOM acceptance and an outer host-write boundary, not a mocked game."""

import numpy as np
import pytest

from doomformer.agent import Agent, Actuator
from doomformer.kit import MLP, MismatchError
from doomformer.readout import ReadoutOnly
from doomformer import sense
from doomformer.onemap import tiny_model


def room(m):
    m.rect(5120, -7168, 7168, -5120, floor=0, ceil=128, tag=20)
    m.thing(6144, -6144, 1)
    return ''


class OuterNoWrites:
    """Independent wrapper outside Agent's audit: no buttons, resets or port writes."""
    def __init__(self, game, allowed):
        self.game, self.allowed, self.commands = game, set(allowed), []

    def send_game_command(self, command):
        assert command in self.allowed, f'outer wrapper caught host write: {command}'
        self.commands.append(command)
        return self.game.send_game_command(command)

    def __getattr__(self, name):
        assert name in {'advance_action', 'get_game_variable', 'get_state', 'get_episode_time',
                        'is_episode_finished', 'close'}, f'outer wrapper caught {name}'
        return getattr(self.game, name)


def net_for(n, outputs=1):
    return MLP([np.ones((outputs, n), dtype=np.int64)],
               [np.zeros(outputs, dtype=np.int64)], [np.array([1])])


def test_all_sensors_and_outer_readonly_boundary():
    sensors = [sense.bearing(), sense.distance(), sense.health(), sense.ammo(),
               sense.x_bin(64), sense.y_bin(64), sense.monster_count(),
               sense.sector_flag(20), sense.sector_flag(21), sense.line_of_sight()]
    with Agent(net_for(len(sensors), 2), sensors=sensors,
               actuators={'y1': Actuator.turn(640), 'y0': Actuator.turn(-640)}, world=room) as agent:
        allowed = agent.game.allowed
        outer = OuterNoWrites(agent.game, allowed)
        agent.game = agent.runner.game = outer
        rows = agent.run(2) + agent.run(1)
        assert [r['step'] for r in rows] == [1, 2, 3]
        assert rows[0]['inputs'] == [17, 4, 100, 50, 96, -96, 0, 1, 0, 0]
        assert all(r['action'] == 'y1' and r['exact'] for r in rows)  # insertion-order tie
        assert rows[1]['sample']['angle'] == 640
        assert outer.commands and all(c in allowed for c in outer.commands)
        # Restore the audit object for audit() and close().
        agent.game = agent.runner.game = outer.game
        assert agent.audit()['non_readout_commands'] == 0



def test_transformer_token_sensors_all_positions():
    net = tiny_model()
    with Agent(net, sensors=[sense.constant(0), sense.constant(1)],
               actuators={'A': Actuator.turn(100), 'B': Actuator.turn(-100)}, world=room) as agent:
        row = agent.run(1)[0]
        assert row['exact']
        assert row['checked'] == len(agent.meta.outputs) + 3
        assert any(name.startswith('logits:2:') for name in row['outputs'])
        assert agent.audit()['non_readout_commands'] == 0


def test_readout_audit_rejects_writes():
    class NeverCalled:
        def send_game_command(self, command):
            raise AssertionError('forbidden command reached the engine')
    guard = ReadoutOnly(NeverCalled(), {'puke -100'})
    for command in ['puke -1 1000 100 0', 'puke -3', 'puke -100; god', 'god']:
        with pytest.raises(RuntimeError, match='non-readout'):
            guard.send_game_command(command)
    for name in ['set_action', 'make_action', 'new_episode']:
        with pytest.raises(RuntimeError, match='rejected'):
            getattr(guard, name)
    assert guard.audit()['non_readout_commands'] == 7


def test_sensor_replica_edges_and_ties():
    def candidate(distance, bearing, sight=1, health=1):
        return [distance, 0, distance, bearing, sight, health]
    assert sense.nearest_replica(0, [])['count'] == 0
    same = [candidate(100 << 16, 0, 0), candidate(100 << 16, -3000, 1),
            candidate(1, 0, 0, 0)]
    nearest = sense.nearest_replica(0, same)
    assert nearest['count'] == 2 and nearest['sight'] == 1
    assert nearest['distance'] == sense.nearest_replica(0, [candidate((100 << 16)-1, 0)])['distance']+1
    assert sense.x_bin(64).replica(-65 << 16, nearest) == -1
    assert sense.health(10, 105).replica(100, nearest) == 0


def test_mutated_readback_is_not_silently_accepted():
    with Agent(net_for(1), sensors=[sense.health()], actuators={0: Actuator.turn(0)}, world=room) as agent:
        read = agent.runner.read
        def corrupt():
            got = read()
            got['y0'] += 1
            return got
        agent.runner.read = corrupt
        with pytest.raises(MismatchError):
            agent.run(1)


def test_world_actuators_and_live_sampling():
    def world(m):
        m.rect(-1024, -3072, 0, -2048, floor=0, ceil=128, tag=20)
        m.rect(0, -2816, 64, -2304, floor=0, ceil=0, tag=21)
        m.rect(64, -2816, 320, -2304, floor=0, ceil=128)
        m.thing(-512, -2560, 1)
        return '''
script 220 (int value) { SetActorProperty(900, APROP_Health, value); }
script 221 OPEN {
  while(GetSectorCeilingZ(21,0,0)<32.0) Delay(1);
  SetActorProperty(900, APROP_Health, 77);
}
'''
    with Agent(net_for(3), sensors=[sense.health(), sense.x_bin(1), sense.sector_flag(20)],
               actuators={0: [Actuator.thrust(0, 4), Actuator.door(21), Actuator.script(220, 88)]},
               world=world, freeze_monsters=False) as agent:
        rows = agent.run(2)
        assert rows[1]['inputs'][0] == 77
        assert rows[1]['inputs'][1] > rows[0]['inputs'][1]
        assert rows[1]['inputs'][2] == 1
        assert agent.audit()['non_readout_commands'] == 0


@pytest.mark.parametrize('blocked', [False, True])
def test_nearest_monster_sample_and_sight(blocked):
    def world(m):
        m.rect(-1024, -3072, -400, -2048, floor=0, ceil=128, tag=20)
        m.rect(-400, -3072, -384, -2048, floor=0, ceil=0 if blocked else 128)
        m.rect(-384, -3072, 0, -2048, floor=0, ceil=128)
        m.thing(-312, -2560, 3002, id=901)
        m.thing(-212, -2560, 3002, id=901)
        m.thing(-512, -2560, 1)
        return ''
    sensors = [sense.bearing(), sense.distance(), sense.monster_count(), sense.line_of_sight()]
    with Agent(net_for(4), sensors=sensors, actuators={0: Actuator.turn(0)}, world=world) as agent:
        row = agent.run(1)[0]
        assert row['inputs'] == [8, 2, 2, int(not blocked)]
        assert len(row['sample']['monsters']) == 2
        assert row['exact']


def test_mlp_hidden_values_and_explicit_period():
    net = MLP([[[1, -1], [-1, 2]], [[1, 2], [2, -1]]],
              [[0, 1], [1, -1]], [[1, 2], [1, 2]])
    with Agent(net, sensors=[sense.constant(2), sense.constant(-1)],
               actuators={0: Actuator.turn(0), 1: Actuator.turn(0)},
               world=room, period=1000) as agent:
        rows = agent.run(2)
        assert all(row['exact'] for row in rows)
        assert any(name.startswith('h1_') for name in rows[0]['outputs'])
        assert any(name.startswith('-h1_') for name in rows[0]['outputs'])


def test_undersized_period_is_detected():
    with Agent(net_for(10), sensors=[sense.constant(1)] * 10,
               actuators={0: Actuator.turn(0)}, world=room, period=1, world_tics=1) as agent:
        with pytest.raises(MismatchError):
            agent.run(1)
