"""A keyboard for `doomformer draw` under Xvfb: real X key events through the XTest extension, screenshots of the
X root window. Plays: draw a 7, READ, wait for the answer, CLEAR, draw a 1, READ, wait, close.

usage (inside the same X server as the game): xdrive.py LOG OUTDIR
LOG: the `doomformer draw` stdout file (the driver waits for its lines the way a player reads the terminal).
"""
import os
import re
import sys
import time

from PIL import Image
from Xlib import X, XK, display
from Xlib.ext import xtest

LOG, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = display.Display()
root = d.screen().root
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


def log():
    with open(LOG, errors="replace") as f:
        return f.read()


def wait_log(pattern, timeout, start=0):
    end = time.time() + timeout
    while time.time() < end:
        m = re.search(pattern, log()[start:])
        if m:
            return m
        time.sleep(0.5)
    raise TimeoutError(f"no {pattern!r} in {LOG} after {timeout} s")


def windows(w=None):
    w = w or root
    for c in w.query_tree().children:
        yield c
        yield from windows(c)


def find_game(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        for w in windows():
            try:
                name = w.get_wm_name()
                attrs = w.get_attributes()
            except Exception:
                continue
            if name and "doom" in name.lower() and attrs.map_state == X.IsViewable:
                return w
        time.sleep(0.5)
    raise TimeoutError("no ViZDoom window")


def shot(name):
    g = root.get_geometry()
    raw = root.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xFFFFFFFF)
    img = Image.frombytes("RGB", (g.width, g.height), raw.data, "raw", "BGRX")
    path = os.path.join(OUT, name)
    img.save(path)
    say(f"screenshot {path}")
    return img


def code(key):
    return d.keysym_to_keycode(XK.string_to_keysym(key))


def down(*keys):
    for k in keys:
        xtest.fake_input(d, X.KeyPress, code(k))
    d.sync()


def up(*keys):
    for k in keys:
        xtest.fake_input(d, X.KeyRelease, code(k))
    d.sync()


def hold(seconds, *keys):
    down(*keys)
    time.sleep(seconds)
    up(*keys)


def tap(key, seconds=0.15):
    hold(seconds, key)


# --- the session -------------------------------------------------------------------------------------------------
def crop_shot(name):
    img = shot("full_" + name)
    g = win.get_geometry()
    pos = win.translate_coords(root, 0, 0)
    x, y = -pos.x, -pos.y
    img.crop((x, y, x + g.width, y + g.height)).save(os.path.join(OUT, name))


def walk(seconds, *keys, settle=1.5):
    hold(seconds, *keys)
    time.sleep(settle)


def pointer_held_by_game():
    st = root.grab_pointer(False, X.ButtonPressMask, X.GrabModeAsync, X.GrabModeAsync, X.NONE, X.NONE, X.CurrentTime)
    if st == 0:
        d.ungrab_pointer(X.CurrentTime)
    d.sync()
    return st == 1                       # AlreadyGrabbed: another client (the game) holds the pointer


wait_log(r"You play in the DOOM window", 300)
say("help printed (the game is running); looking for the window")
win = find_game()
g = win.get_geometry()
pos = win.translate_coords(root, 0, 0)
say(f"window {win.get_wm_name()!r} {g.width}x{g.height} at {-pos.x},{-pos.y}")
time.sleep(1)
# a player clicks the window: pointer onto it, one left click (FIRE off the canvas paints nothing)
xtest.fake_input(d, X.MotionNotify, x=-pos.x + g.width // 2, y=-pos.y + g.height // 2)
d.sync()
time.sleep(0.3)
xtest.fake_input(d, X.ButtonPress, 1)
d.sync()
time.sleep(0.1)
xtest.fake_input(d, X.ButtonRelease, 1)
d.sync()
time.sleep(1)
say(f"input focus: {d.get_input_focus().focus}; pointer held by the game: {pointer_held_by_game()}")
crop_shot("00_start.png")

# --- a 7 with the arrow keys, Ctrl and , . -------------------------------------------------------------------------
say("7: walk north to the top-left of the 7")
walk(2.85, "Up")
walk(0.6, "comma")
say("7: Ctrl held, top bar (strafe right)")
down("Control_L")
walk(1.4, "period", settle=0.8)
say("7: Ctrl held, diagonal down-left (Down held, comma pulsed)")
down("Down")
for _ in range(3):
    hold(0.35, "comma")
    time.sleep(0.25)
crop_shot("01_drawing.png")
time.sleep(max(0.0, 1.9 - 3 * 0.6))
up("Down")
time.sleep(0.8)
up("Control_L")
time.sleep(1.0)
crop_shot("01b_drawn.png")
say("to READ: down and right into the south-east corner (walls stop him), left 432 units, one step north")
walk(4.0, "Down", "period")
walk(1.53, "comma")
walk(0.3, "Up")
mark = len(log())
tap("space")
time.sleep(2)
crop_shot("02_read_pressed.png")
wait_log(r"READ 1: DOOM is reading", 30, mark)
say("CLI: READ 1 started")
for k in range(4):
    time.sleep(1.5)
    crop_shot(f"02b_thinking{k}.png")
m = wait_log(r"DOOM reads: (\d)", 600, mark)
say(f"CLI: DOOM reads {m.group(1)}")
time.sleep(1)
crop_shot("03_answer.png")
wait_log(r"CLEAR to draw another digit", 60, mark)
say(f"pointer held by the game: {pointer_held_by_game()}; press Esc (DOOM's menu)")
tap("Escape")
time.sleep(1)
crop_shot("03b_menu.png")
say(f"menu open: pointer held by the game: {pointer_held_by_game()}; Esc again closes the menu")
tap("Escape")
time.sleep(1)
say(f"menu closed: pointer held by the game: {pointer_held_by_game()}")

# --- CLEAR with E, then a 1 with W/A/S/D and Ctrl ------------------------------------------------------------------
say("to CLEAR: into the south-west corner, right 432 units, one step north, E")
walk(4.0, "s", "a")
walk(1.53, "d")
walk(0.3, "w")
mark = len(log())
tap("e")
time.sleep(1.5)
crop_shot("04_cleared.png")
say("1: south-west corner, right 260, north 864 (west of the podiums), right 316, then Ctrl + S south")
walk(3.0, "s", "a")
walk(0.9, "d")
walk(3.1, "w")
walk(1.1, "d")
down("Control_L")
hold(1.0, "s")
crop_shot("05_drawing.png")
hold(0.8, "s")
time.sleep(0.8)
up("Control_L")
time.sleep(1.0)
crop_shot("05b_drawn.png")
say("to READ: south-east corner, left 432, one step north, Space")
walk(4.0, "s", "d")
walk(1.53, "a")
walk(0.3, "w")
mark = len(log())
tap("space")
wait_log(r"READ 2: DOOM is reading", 30, mark)
say("CLI: READ 2 started; walk to CLEAR at once and press E (about 6 s into an 8 s READ)")
walk(4.0, "s", "a")
walk(1.53, "d")
walk(0.3, "w")
tap("e")
time.sleep(0.7)
crop_shot("05c_clear_refused.png")
m = wait_log(r"DOOM reads: (\d)", 600, mark)
say(f"CLI: DOOM reads {m.group(1)}")
time.sleep(1)
crop_shot("06_answer.png")
wait_log(r"CLEAR to draw another digit", 60, mark)

# --- quit: close the window the way a window manager's close button does -------------------------------------------
say("closing the window (WM_DELETE_WINDOW)")
from Xlib.protocol import event  # noqa: E402
proto, delete = d.intern_atom("WM_PROTOCOLS"), d.intern_atom("WM_DELETE_WINDOW")
win.send_event(event.ClientMessage(window=win, client_type=proto, data=(32, [delete, X.CurrentTime, 0, 0, 0])))
d.sync()
wait_log(r"the DOOM window closed", 60)
say("CLI: the DOOM window closed")
