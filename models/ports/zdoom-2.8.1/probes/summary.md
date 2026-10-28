### zdoom-2.8.1: probes: 51 PASS, 0 DIFF, 0 ERROR of 51 WADs

| row | expected | observed | result | s |
|---|---|---|---|---|
| light_saturates | `light_saturates up1=30160 up2=32767 down=-32768` | `light_saturates up1=30160 up2=32767 down=-32768` | PASS | 4.1 |
| light_same_tic | `light_same_tic light=560` | `light_same_tic light=560` | PASS | 4.1 |
| mover_same_tic | `mover_same_tic first100=100 first50=50` | `mover_same_tic first100=100 first50=50` | PASS | 5.5 |
| mover_active_lock | `mover_active_lock busy=100 after=50 floor=100 ceil=1800` | `mover_active_lock busy=100 after=50 floor=100 ceil=1800` | PASS | 7.3 |
| mover_speed | `12 values in 1 lines (expected.txt)` | `12 values, byte for byte equal` | PASS | 4.4 |
| mover_busy | `mover_busy fr_14018=-243 fr_14019=12 fl_m14017=243 fl_m14018=-12 cr_14017=-243 cr_14018=12 cl_m14018=243 cl_m14019=-12 fr_14019_tic3=-243` | `mover_busy fr_14018=-243 fr_14019=12 fl_m14017=243 fl_m14018=-12 cr_14017=-243 cr_14018=12 cl_m14018=243 cl_m14019=-12 fr_14019_tic3=-243` | PASS | 1.8 |
| plane_span_32767 | `plane_span_32767 span=32767 moved=1` | `plane_span_32767 span=32767 moved=1` | PASS | 2.5 |
| plane_span_32768 | `plane_span_32768 span=32768 moved=0` | `plane_span_32768 span=32768 moved=0` | PASS | 2.5 |
| plane_span_32000 | `plane_span_32000 span=32000 moved=1` | `plane_span_32000 span=32000 moved=1` | PASS | 2.5 |
| plane_span_33000 | `plane_span_33000 span=33000 moved=0` | `plane_span_33000 span=33000 moved=0` | PASS | 2.5 |
| voodoo_tid | `voodoo_tid used101=0 used102=0 used103=0 used104=0` | `voodoo_tid used101=0 used102=0 used103=0 used104=0` | PASS | 1.5 |
| carry_stack | `carry_stack d1=78 d2=157 d3=236 d4=315` | `carry_stack d1=78 d2=157 d3=236 d4=315` | PASS | 3.2 |
| setcurrent_doll | `setcurrent_doll doll_current=0 doll_carry=1` | `setcurrent_doll doll_current=0 doll_carry=1` | PASS | 3.6 |
| setcurrent_player_current | `setcurrent_player_current player_current=0` | `setcurrent_player_current player_current=0` | PASS | 2.1 |
| setcurrent_player_carry | `setcurrent_player_carry player_carry=1` | `setcurrent_player_carry player_carry=1` | PASS | 2.0 |
| setcurrent_barrel | `setcurrent_barrel barrel=0` | `setcurrent_barrel barrel=0` | PASS | 2.5 |
| doll_steps | `doll_steps step24=1 step25=0 gap56=1 gap55=0` | `doll_steps step24=1 step25=0 gap56=1 gap55=0` | PASS | 3.8 |
| gate_opens | `gate_opens opens0=1 opens1=0 opens10=0 opens24=0 arrives10=1 arrives24=1` | `gate_opens opens0=1 opens1=0 opens10=0 opens24=0 arrives10=1 arrives24=1` | PASS | 5.5 |
| self_tag | `self_tag crossed=1 idle=0 east=1` | `self_tag crossed=1 idle=0 east=1` | PASS | 2.3 |
| line_special_tid_player | `line_special_tid_player player_moved=1 doll_pushed=0` | `line_special_tid_player player_moved=1 doll_pushed=0` | PASS | 3.8 |
| line_special_tid_doll | `line_special_tid_doll player_moved=0 doll_pushed=1` | `line_special_tid_doll player_moved=0 doll_pushed=1` | PASS | 3.8 |
| line_order_teleport_last_4 | `line_order_teleport_last_4 same=10 bad=0` | `line_order_teleport_last_4 same=10 bad=0` | PASS | 4.3 |
| line_order_teleport_last_8 | `line_order_teleport_last_8 same=4 bad=0` | `line_order_teleport_last_8 same=4 bad=0` | PASS | 4.4 |
| line_order_teleport_last_64 | `line_order_teleport_last_64 same=0 bad=0` | `line_order_teleport_last_64 same=0 bad=0` | PASS | 4.4 |
| line_order_teleport_first_4 | `line_order_teleport_first_4 same=10 bad=0` | `line_order_teleport_first_4 same=10 bad=0` | PASS | 4.3 |
| line_order_teleport_first_8 | `line_order_teleport_first_8 same=4 bad=0` | `line_order_teleport_first_8 same=4 bad=0` | PASS | 4.3 |
| line_order_teleport_first_64 | `line_order_teleport_first_64 same=0 bad=0` | `line_order_teleport_first_64 same=0 bad=0` | PASS | 4.3 |
| wrap_lone_room_-30000 | `wrap_lone_room_-30000 floor=64` | `wrap_lone_room_-30000 floor=64` | PASS | 1.5 |
| wrap_lone_room_-32000 | `wrap_lone_room_-32000 floor=0` | `wrap_lone_room_-32000 floor=0` | PASS | 1.5 |
| wrap_lone_room_28000 | `wrap_lone_room_28000 floor=64` | `wrap_lone_room_28000 floor=64` | PASS | 1.4 |
| wrap_lone_room_32000 | `wrap_lone_room_32000 floor=64` | `wrap_lone_room_32000 floor=64` | PASS | 1.5 |
| wrap_lone_room_32700 | `wrap_lone_room_32700 floor=64` | `wrap_lone_room_32700 floor=64` | PASS | 1.5 |
| wrap_lone_room_32800 | `wrap_lone_room_32800 floor=0` | `wrap_lone_room_32800 floor=0` | PASS | 1.5 |
| wrap_lone_room_33000 | `wrap_lone_room_33000 floor=0` | `wrap_lone_room_33000 floor=0` | PASS | 1.5 |
| wrap_lone_room_34000 | `wrap_lone_room_34000 floor=0` | `wrap_lone_room_34000 floor=0` | PASS | 1.5 |
| wrap_lone_room_36000 | `wrap_lone_room_36000 floor=64` | `wrap_lone_room_36000 floor=64` | PASS | 1.5 |
| wrap_balanced_x_60000 | `wrap_balanced_x_60000 bad=0` | `wrap_balanced_x_60000 bad=0` | PASS | 1.5 |
| wrap_balanced_y_60000 | `wrap_balanced_y_60000 bad=0` | `wrap_balanced_y_60000 bad=0` | PASS | 1.5 |
| wrap_balanced_x_64000 | `wrap_balanced_x_64000 bad=0` | `wrap_balanced_x_64000 bad=0` | PASS | 1.5 |
| wrap_balanced_y_64000 | `wrap_balanced_y_64000 bad=0` | `wrap_balanced_y_64000 bad=0` | PASS | 1.5 |
| wrap_balanced_y_65000 | `wrap_balanced_y_65000 bad=1` | `wrap_balanced_y_65000 bad=1` | PASS | 1.5 |
| doll_fall | `doll_fall below1=-2 below3=-9 below6=-27 fall_tics=10 landed=-70` | `doll_fall below1=-2 below3=-9 below6=-27 fall_tics=10 landed=-70` | PASS | 3.2 |
| dormant | `dormant start_dormant=60 awake=40 activated=40 deactivated=40` | `dormant start_dormant=60 awake=40 activated=40 deactivated=40` | PASS | 1.9 |
| wrap_shifted_26000 | `wrap_shifted_26000 floor=64` | `wrap_shifted_26000 floor=64` | PASS | 1.5 |
| wrap_shifted_28000 | `wrap_shifted_28000 floor=64` | `wrap_shifted_28000 floor=64` | PASS | 1.5 |
| wrap_shifted_30000 | `wrap_shifted_30000 floor=0` | `wrap_shifted_30000 floor=0` | PASS | 1.5 |
| wrap_shifted_31000 | `wrap_shifted_31000 floor=0` | `wrap_shifted_31000 floor=0` | PASS | 1.5 |
| wrap_shifted_32000 | `wrap_shifted_32000 floor=0` | `wrap_shifted_32000 floor=0` | PASS | 1.5 |
| line_floor_speed | `line_floor_speed s150000=0 s255=0 s240=0 s100=0 s0=24` | `line_floor_speed s150000=0 s255=0 s240=0 s100=0 s0=24` | PASS | 3.8 |
| tag_range | `tag_range t1000=8 t20000=16 t32767=24 t32768=32 t40000=40 t60000=48 t64004=56 t65535=64 line64010=-8` | `tag_range t1000=8 t20000=16 t32767=24 t32768=32 t40000=40 t60000=48 t64004=56 t65535=64 line64010=-8` | PASS | 3.8 |
| tag_wrap | `tag_wrap t40000=8 tm25536=0 t65541=16 t5=0` | `tag_wrap t40000=8 tm25536=0 t65541=16 t5=0` | PASS | 1.5 |
