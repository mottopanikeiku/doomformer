| row | gzdoom-4.14.2-opengl | gzdoom-4.14.2-vulkan | lzdoom | uzdoom | zandronum-3.2.1 | zdoom-2.8.1 |
|---|---|---|---|---|---|---|
| light_saturates | PASS | PASS | PASS | PASS | PASS | PASS |
| light_same_tic | PASS | PASS | PASS | PASS | PASS | PASS |
| mover_same_tic | PASS | PASS | PASS | PASS | PASS | PASS |
| mover_active_lock | PASS | PASS | PASS | PASS | PASS | PASS |
| mover_speed | DIFF t1_300000=12000 (want 9464) t1_500000=12000 (want -6072) t3_500000=12000 (want -12144) | DIFF t1_300000=12000 (want 9464) t1_500000=12000 (want -6072) t3_500000=12000 (want -12144) | DIFF t1_300000=12000 (want 9464) t1_500000=12000 (want -6072) t3_500000=12000 (want -12144) | DIFF t1_300000=12000 (want 9464) t1_500000=12000 (want -6072) t3_500000=12000 (want -12144) | PASS | PASS |
| mover_busy | DIFF fr_14019=-243 (want 12) fl_m14018=243 (want -12) cr_14018=-243 (want 12) cl_m14019=243 (want -12) | DIFF fr_14019=-243 (want 12) fl_m14018=243 (want -12) cr_14018=-243 (want 12) cl_m14019=243 (want -12) | DIFF fr_14019=-243 (want 12) fl_m14018=243 (want -12) cr_14018=-243 (want 12) cl_m14019=243 (want -12) | DIFF fr_14019=-243 (want 12) fl_m14018=243 (want -12) cr_14018=-243 (want 12) cl_m14019=243 (want -12) | PASS | PASS |
| plane_span_32767 | PASS | PASS | PASS | PASS | PASS | PASS |
| plane_span_32768 | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | PASS | PASS |
| plane_span_32000 | PASS | PASS | PASS | PASS | PASS | PASS |
| plane_span_33000 | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | DIFF moved=1 (want 0) | PASS | PASS |
| voodoo_tid | PASS | PASS | PASS | PASS | PASS | PASS |
| carry_stack | PASS | PASS | PASS | PASS | PASS | PASS |
| setcurrent_doll | PASS | PASS | PASS | PASS | PASS | PASS |
| setcurrent_player_current | PASS | PASS | PASS | PASS | PASS | PASS |
| setcurrent_player_carry | PASS | PASS | PASS | PASS | PASS | PASS |
| setcurrent_barrel | PASS | PASS | PASS | PASS | PASS | PASS |
| doll_steps | PASS | PASS | PASS | PASS | PASS | PASS |
| gate_opens | PASS | PASS | PASS | PASS | PASS | PASS |
| self_tag | PASS | PASS | PASS | PASS | PASS | PASS |
| line_special_tid_player | PASS | PASS | PASS | PASS | PASS | PASS |
| line_special_tid_doll | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_last_4 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_last_8 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_last_64 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_first_4 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_first_8 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_order_teleport_first_64 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_lone_room_-30000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_lone_room_-32000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_lone_room_28000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_lone_room_32000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_lone_room_32700 | ERROR refused: Value 32956.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32956.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32956.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32956.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | PASS | PASS |
| wrap_lone_room_32800 | ERROR refused: Value 32800.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32800.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32800.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 32800.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | PASS | PASS |
| wrap_lone_room_33000 | ERROR refused: Value 33000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 33000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 33000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 33000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | PASS | PASS |
| wrap_lone_room_34000 | ERROR refused: Value 34000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 34000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 34000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 34000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | PASS | PASS |
| wrap_lone_room_36000 | ERROR refused: Value 36000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 36000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 36000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | ERROR refused: Value 36000.000000 out of range for a coordinate 'X'. Valid range is [-32768 .. 32768] | PASS | PASS |
| wrap_balanced_x_60000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_balanced_y_60000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_balanced_x_64000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_balanced_y_64000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_balanced_y_65000 | PASS | PASS | PASS | PASS | PASS | PASS |
| doll_fall | PASS | PASS | PASS | PASS | PASS | PASS |
| dormant | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_shifted_26000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_shifted_28000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_shifted_30000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_shifted_31000 | PASS | PASS | PASS | PASS | PASS | PASS |
| wrap_shifted_32000 | PASS | PASS | PASS | PASS | PASS | PASS |
| line_floor_speed | PASS | PASS | PASS | PASS | PASS | PASS |
| tag_range | PASS | PASS | PASS | PASS | DIFF t32768=0 (want 32) t40000=0 (want 40) t60000=0 (want 48) t64004=0 (want 56) t65535=0 (want 64) line64010=0 (want -8) | PASS |
| tag_wrap | PASS | PASS | PASS | PASS | DIFF t40000=0 (want 8) tm25536=8 (want 0) t65541=0 (want 16) t5=16 (want 0) | PASS |
| same_tic_cell4 | PASS | PASS | PASS | PASS | DIFF far=11 (want 14) | PASS |
| same_tic_cell16 | PASS | PASS | PASS | PASS | DIFF far=11 (want 0) | PASS |
| same_tic_cell4_tags15 | PASS | PASS | PASS | PASS | PASS | PASS |
| same_tic_cell16_tags15 | PASS | PASS | PASS | PASS | PASS | PASS |
| hello | PASS | PASS | PASS | PASS | DIFF (5 of 5 lines printed, first difference at line 1) | PASS |
| hello_tags15 | PASS | PASS | PASS | PASS | PASS | PASS |
| machine_multiply | PASS | PASS | PASS | PASS | - | - |
| machine_subleq_add | PASS | PASS | PASS | PASS | DIFF (2 of 5 lines printed, first difference at line 2) | PASS |
| machine_multiply_2_3 | PASS | PASS | PASS | PASS | DIFF (2 of 51 lines printed, first difference at line 2) | PASS |
| machine_subleq_add_tags15 | - | - | - | - | PASS | - |
| machine_multiply_2_3_tags15 | - | - | - | - | PASS | - |

* gzdoom-4.14.2-opengl: 51 PASS, 4 DIFF, 5 ERROR of 60
* gzdoom-4.14.2-vulkan: 51 PASS, 4 DIFF, 5 ERROR of 60
* lzdoom: 51 PASS, 4 DIFF, 5 ERROR of 60
* uzdoom: 51 PASS, 4 DIFF, 5 ERROR of 60
* zandronum-3.2.1: 54 PASS, 7 DIFF, 0 ERROR of 61
* zdoom-2.8.1: 59 PASS, 0 DIFF, 0 ERROR of 59
