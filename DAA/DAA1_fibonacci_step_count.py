"""
===============================================================================
DAA1 : Write a program to calculate Fibonacci numbers and find its step count.
       (Syllabus: non-recursive and recursive program, analyze time & space)
===============================================================================

THEORY
------
1. ALGORITHM (Unit I)
   An algorithm is a finite sequence of well-defined, unambiguous instructions
   that takes some input and produces output in finite time. Properties:
     - Input       : zero or more quantities supplied externally.
     - Output      : at least one quantity is produced.
     - Definiteness: every instruction is clear and unambiguous.
     - Finiteness  : terminates after a finite number of steps.
     - Effectiveness: every step is basic enough to be carried out on paper.
   Correctness of an algorithm is usually proved with a LOOP INVARIANT
   (initialization, maintenance, termination) for iterative algorithms and
   with MATHEMATICAL INDUCTION for recursive ones.

2. FIBONACCI SERIES
       F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)   for n >= 2
       0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
   Closed form (Binet): F(n) = (phi^n - psi^n) / sqrt(5),
       phi = (1 + sqrt5)/2 ~ 1.618 (golden ratio), psi = (1 - sqrt5)/2.

3. STEP COUNT METHOD (Unit II - "Counting dominant operators")
   Time complexity can be measured by counting the number of "program steps".
   A program step is a syntactically/semantically meaningful segment whose
   execution time is independent of the instance characteristics, e.g.
       - assignment / comparison / arithmetic / return  -> 1 step
       - a loop header executes (n + 1) times (last test fails)
       - comments and declarations                       -> 0 steps
   We can either build a "frequency table" (steps/execution x frequency) or
   insert a global counter `count` into the program (done below).
   The DOMINANT OPERATION (here: the addition) decides the growth rate.

4. ITERATIVE (NON-RECURSIVE) FIBONACCI
       a = 0, b = 1
       repeat n-1 times: c = a + b; a = b; b = c
   Loop invariant: before the i-th iteration a = F(i-2), b = F(i-1).
   Steps ~ 4n + c  ->  Time  T(n) = O(n),  Space S(n) = O(1).

5. RECURSIVE FIBONACCI
       fib(n) = n                     if n <= 1
              = fib(n-1) + fib(n-2)   otherwise
   Recurrence: T(n) = T(n-1) + T(n-2) + c,  T(0) = T(1) = c
   The recursion tree has F(n+1) leaves; number of calls = 2F(n+1) - 1.
   Hence T(n) = Theta(phi^n) ~ O(1.618^n), loosely written O(2^n)  (EXPONENTIAL).
   Space = depth of recursion stack = O(n).
   It is slow because the same sub-problems are recomputed many times
   (OVERLAPPING SUB-PROBLEMS). Storing answers (memoization / tabulation)
   turns it into Dynamic Programming with O(n) time - see Unit III.

6. ASYMPTOTIC NOTATIONS (Unit II)
   - Big-O   (upper bound) : f(n) = O(g(n)) if f(n) <= c.g(n) for all n >= n0
   - Omega   (lower bound) : f(n) = Omega(g(n)) if f(n) >= c.g(n) for n >= n0
   - Theta   (tight bound) : c1.g(n) <= f(n) <= c2.g(n) for n >= n0
   - little-o / little-omega: strict upper / strict lower bounds.
   Common growth classes: 1 < log n < n < n log n < n^2 < n^3 < 2^n < n!

7. BEST / WORST / AVERAGE CASE
   For Fibonacci the work depends only on n (not on the data arrangement), so
   best = worst = average case. Input size is the number of bits of n
   (log n), which is why even O(n) Fibonacci is "pseudo-polynomial".

8. SPACE COMPLEXITY
   S(P) = C (fixed part: code, constants, simple variables) + Sp (variable part:
   recursion stack, dynamic arrays). Iterative: O(1). Recursive: O(n) stack.

COMPARISON
   | Method     | Time      | Space | Remark                         |
   |------------|-----------|-------|--------------------------------|
   | Iterative  | O(n)      | O(1)  | efficient                      |
   | Recursive  | O(phi^n)  | O(n)  | recomputes sub-problems        |
   | Memoized   | O(n)      | O(n)  | DP (top-down)                  |

CONCLUSION: The iterative version is linear while the naive recursive version
is exponential; the step counts printed below demonstrate this.
===============================================================================
"""

import sys

# ---------------------------------------------------------------------------
# 1. Iterative (non-recursive) Fibonacci with step count
# ---------------------------------------------------------------------------
def fib_iterative(n):
    steps = 0
    if n <= 1:                 # 1 comparison
        steps += 2             # comparison + return
        return n, steps
    a, b = 0, 1                # 1 step
    steps += 2                 # comparison above + this assignment
    for _ in range(2, n + 1):  # loop test: (n-1) true + 1 false
        steps += 1             # loop test
        c = a + b              # 1 step (dominant operation)
        a = b                  # 1 step
        b = c                  # 1 step
        steps += 3
    steps += 1                 # final failing loop test
    steps += 1                 # return
    return b, steps


# ---------------------------------------------------------------------------
# 2. Recursive Fibonacci with step count (global counter technique)
# ---------------------------------------------------------------------------
rec_steps = 0
rec_calls = 0

def fib_recursive(n):
    global rec_steps, rec_calls
    rec_calls += 1
    rec_steps += 1             # comparison n <= 1
    if n <= 1:
        rec_steps += 1         # return
        return n
    rec_steps += 2             # addition + return
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# ---------------------------------------------------------------------------
# 3. Memoized (top-down DP) Fibonacci - shows effect of removing overlap
# ---------------------------------------------------------------------------
memo_steps = 0

def fib_memo(n, memo=None):
    global memo_steps
    if memo is None:
        memo = {}
    memo_steps += 1            # lookup / comparison
    if n in memo:
        return memo[n]
    if n <= 1:
        memo_steps += 1
        memo[n] = n
        return n
    memo_steps += 2            # addition + store
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def print_series(n):
    a, b = 0, 1
    series = []
    for _ in range(n):
        series.append(a)
        a, b = b, a + b
    return series


def main():
    global rec_steps, rec_calls, memo_steps
    sys.setrecursionlimit(10000)
    try:
        n = int(input("Enter n (term number, e.g. 10): "))
    except Exception:  # bad input, no input, or notebook without stdin
        n = 10
        print(f"Invalid/no input, using n = {n}")
    if n < 0:
        print("n must be non-negative")
        return

    print(f"\nFirst {n} Fibonacci numbers: {print_series(n)}")

    val, steps = fib_iterative(n)
    print(f"\n[Iterative]  F({n}) = {val:<10} step count = {steps}")

    if n <= 30:
        rec_steps = rec_calls = 0
        val = fib_recursive(n)
        print(f"[Recursive]  F({n}) = {val:<10} step count = {rec_steps}"
              f"  (function calls = {rec_calls})")
    else:
        print("[Recursive]  skipped for n > 30 (exponential time)")

    memo_steps = 0
    val = fib_memo(n)
    print(f"[Memoized ]  F({n}) = {val:<10} step count = {memo_steps}")

    # Table showing growth of step count -> linear vs exponential
    print("\nGrowth of step count")
    print(f"{'n':>4} | {'Iterative':>10} | {'Recursive':>10} | {'Calls':>8}")
    print("-" * 42)
    for k in [0, 1, 2, 5, 10, 15, 20, 25]:
        _, it = fib_iterative(k)
        rec_steps = rec_calls = 0
        fib_recursive(k)
        print(f"{k:>4} | {it:>10} | {rec_steps:>10} | {rec_calls:>8}")

    print("\nTime complexity : Iterative O(n), Recursive O(phi^n) ~ O(2^n)")
    print("Space complexity: Iterative O(1), Recursive O(n) (stack depth)")


if __name__ == "__main__":
    main()
