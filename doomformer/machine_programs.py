"""Ready programs for `doomformer machine` (assembly for doomformer/machine.py). Each entry builds a `Machine` for the
given arguments and says which register holds the answer and what Python computes to compare it with.

Sizes are what a run costs in DOOM, not what the machine can hold: every step is one pass of the level's loop (about
200 to 600 game tics), so the step count decides the wall time. `doomformer machine PROGRAM ARGS` prints the
interpreter's step count before it starts the game."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd as _gcd
from typing import Callable

from .machine import DEFAULT_BASE, Machine

ADD = """
loop: jzdec b done
      inc a
      jmp loop
done: halt"""

MULTIPLY = """
outer:   jzdec a end          ; for each unit of a
inner:   jzdec b restore      ;   c += b, keeping b in t
         inc c
         inc t
         jmp inner
restore: jzdec t outer        ;   b = t
         inc b
         jmp restore
end:     halt"""

POWER = """
pow:   jzdec n done           ; c = c * a, n times
mulc:  jzdec c mdone          ; d = c * a, consuming c, keeping a in t
addup: jzdec a rest
       inc d
       inc t
       jmp addup
rest:  jzdec t mulc
       inc a
       jmp rest
mdone: move d c
       jmp pow
done:  halt"""

GCD = """
outer: jzdec b fin            ; b == 0: the answer is a
       inc b
lp:    jzdec b bz             ; try a -= b, counting the part taken in t
       jzdec a fail
       inc t
       jmp lp
bz:    move t b               ; a -= b done: b restored, t = 0
       jmp lp
fail:  inc a                  ; a < b: (a, b) = (b, a mod b)
       move b a
cp:    jzdec t outer
       inc a
       inc b
       jmp cp
fin:   halt"""

FIBONACCI = """
loop: jzdec c done
      bigadd a b              ; a = a + b
      jzdec c done
      bigadd b a              ; b = a + b
      jmp loop
done: halt"""

COLLATZ = """
loop:  jge n 2 go             ; stop at n == 1
       jmp done
go:    jzdec n even           ; h = n div 2, parity from where n runs out
       jzdec n odd
       inc h
       jmp go
even:  move h n
       inc c
       jmp loop
odd:   add n 4                ; n = 3 * (2h + 1) + 1 = 6h + 4
tri:   jzdec h oend
       add n 6
       jmp tri
oend:  inc c
       jmp loop
done:  halt"""

# primes: count the primes in 2..N. m runs from N down to 2; trial division by d = 2 .. m - 1 (e counts the tries left);
# m mod d by repeated subtraction: x = m, y = d (the register d itself, restored at the end of each try), t = part taken
PRIMES = """
outer: jlt m 2 done
       copy m e t             ; e = m - 2 tries
       add e -2
       add d 2                ; d = 2
try:   jzdec e prime          ; every d tried: m is prime
       copy m x t             ; x = m
lp:    jzdec d bz
       jzdec x fail
       inc t
       jmp lp
bz:    move t d               ; x -= d done, d restored
       jmp lp
fail:  jzdec t comp           ; x ran out: m mod d = t. zero: d divides m
       add d 3                ; d = d' + t + 2 restored, + 1: the next d
       move t d
       jmp try
comp:  clear d
       clear e
       jmp nextm
prime: inc c
       clear d
nextm: add m -1
       jmp outer
done:  halt"""


@dataclass
class Spec:
    name: str
    blurb: str
    args: list[str]
    build: Callable[..., tuple]        # (*int args) -> (machine, answer register, expected int, text)


def _int_args(args, n):
    if len(args) != n:
        raise ValueError(f"takes {n} integer argument(s)")
    return [int(a) for a in args]


def add(a, b):
    return Machine(ADD, registers={"a": a, "b": b}), "a", a + b, f"{a} + {b}"


def multiply(a, b):
    return Machine(MULTIPLY, registers={"a": a, "b": b, "c": 0, "t": 0}), "c", a * b, f"{a} * {b}"


def power(a, n):
    return (Machine(POWER, registers={"a": a, "n": n, "c": 1, "d": 0, "t": 0}), "c", a ** n, f"{a} ^ {n}")


def gcd(a, b):
    if a < 1 or b < 0:
        raise ValueError("gcd needs a >= 1, b >= 0")
    return Machine(GCD, registers={"a": a, "b": b, "t": 0}), "a", _gcd(a, b), f"gcd({a}, {b})"


def fibonacci(n, base=DEFAULT_BASE):
    limbs = 1
    f = [0, 1]
    while len(f) <= n + 1:
        f.append(f[-1] + f[-2])
    while base ** limbs <= f[n + 1]:
        limbs += 1
    m = Machine(FIBONACCI, registers={"a": 0, "b": 1, "c": n}, limbs={"a": limbs, "b": limbs}, base=base)
    return m, ("a" if n % 2 == 0 else "b"), f[n], f"fib({n}) in {limbs} base-{base} limbs"


def collatz(n):
    steps, x = 0, n
    while x != 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        steps += 1
    return (Machine(COLLATZ, registers={"n": n, "h": 0, "c": 0}), "c", steps, f"collatz steps of {n}")


def primes(n):
    count = sum(all(m % d for d in range(2, m)) for m in range(2, n + 1))
    return (Machine(PRIMES, registers={"m": n, "e": 0, "d": 0, "x": 0, "t": 0, "c": 0}), "c", count,
            f"primes in 2..{n}")


def double(n, base=DEFAULT_BASE):
    """2^n with limbs: n doublings of a multi-limb register, a += a."""
    limbs = 1
    while base ** limbs <= 2 ** n:
        limbs += 1
    src = """
loop: jzdec c done
      bigadd a a
      jmp loop
done: halt"""
    return (Machine(src, registers={"a": 1, "c": n}, limbs={"a": limbs}, base=base), "a", 2 ** n,
            f"2 ^ {n} in {limbs} base-{base} limbs")


PROGRAMS = {
    "add": Spec("add", "a + b by moving b into a one unit per step", ["a", "b"], add),
    "multiply": Spec("multiply", "a * b by nested loops", ["a", "b"], multiply),
    "power": Spec("power", "a ^ n by repeated multiplication", ["a", "n"], power),
    "gcd": Spec("gcd", "Euclid's algorithm with mod by repeated subtraction", ["a", "b"], gcd),
    "fib": Spec("fib", "Fibonacci(n) in multi-limb registers (limbs added by addr, carries by jlt)", ["n"], fibonacci),
    "collatz": Spec("collatz", "number of Collatz steps from n (halving and 3n+1 by loops)", ["n"], collatz),
    "primes": Spec("primes", "count the primes in 2..N by trial division (mod by repeated subtraction)", ["N"], primes),
    "pow2": Spec("pow2", "2^n by n doublings of a multi-limb register", ["n"], double),
}
