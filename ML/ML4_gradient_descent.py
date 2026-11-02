"""
===============================================================================
ML4 : Implement Gradient Descent Algorithm to find the local minima of a
      function. For example, find the local minima of the function
      y = (x + 3)^2 starting from the point x = 2.
===============================================================================

THEORY
------
1. OPTIMISATION IN ML (Unit III)
   Training a model = finding parameters that minimise a COST (loss) function
   J(theta), e.g. MSE for regression. When a closed-form solution is costly or
   unavailable, iterative methods such as GRADIENT DESCENT are used.

2. GRADIENT DESCENT
   The gradient dJ/dtheta points in the direction of steepest INCREASE of J.
   Moving opposite to it decreases J:
        theta_new = theta_old - alpha * dJ/dtheta
   alpha = LEARNING RATE (step size).
   Repeat until the change |theta_new - theta_old| < precision (tolerance) or
   max iterations reached.

3. FOR y = (x + 3)^2
        dy/dx = 2(x + 3)
        x_new = x - alpha * 2(x + 3)
   Analytical minimum: dy/dx = 0 -> x = -3, y = 0.
   Starting at x = 2 with alpha = 0.01, x approaches -3.
   Since x_new + 3 = (1 - 2 alpha)(x + 3), the error shrinks by factor
   (1 - 2 alpha) each step -> converges if 0 < alpha < 1.

4. EFFECT OF LEARNING RATE
   - too small : very slow convergence (many iterations)
   - too large : overshoots, oscillates, may DIVERGE
   - suitable  : smooth fast convergence
   For non-convex functions GD can get stuck in a LOCAL minimum or saddle
   point; result depends on starting point.

5. VARIANTS
   - Batch GD      : gradient using all m training samples per update.
   - Stochastic GD : one random sample per update (noisy, fast).
   - Mini-batch GD : small batch (e.g. 32) - used in deep learning.
   - Momentum, RMSProp, Adam : adaptive improvements.

6. GD FOR LINEAR REGRESSION
        J(b0,b1) = (1/2m) SUM (b0 + b1 x_i - y_i)^2
        b0 := b0 - alpha (1/m) SUM (y_hat_i - y_i)
        b1 := b1 - alpha (1/m) SUM (y_hat_i - y_i) x_i
   Lasso / Ridge add penalty gradients (L1: lambda sign(b), L2: 2 lambda b).

CONCLUSION: Gradient descent starting from x = 2 converges to the local
(and global) minimum x = -3, y = 0 of y = (x+3)^2.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x + 3) ** 2


def df(x):
    return 2 * (x + 3)


def gradient_descent(x0, alpha=0.01, precision=1e-6, max_iters=10000):
    x = x0
    history = [x]
    for i in range(1, max_iters + 1):
        prev = x
        x = x - alpha * df(prev)
        history.append(x)
        if abs(x - prev) < precision:
            break
    return x, i, history


def main():
    try:
        x0 = float(input("Starting point x0 (Enter for 2): ") or 2)
        alpha = float(input("Learning rate (Enter for 0.01): ") or 0.01)
    except Exception:  # bad input, no input, or notebook without stdin
        x0, alpha = 2.0, 0.01
    x_min, iters, hist = gradient_descent(x0, alpha)

    print("\nFirst 10 iterations:")
    print(f"{'Iter':>5} {'x':>12} {'y=(x+3)^2':>14} {'gradient':>12}")
    for i, x in enumerate(hist[:11]):
        print(f"{i:>5} {x:>12.6f} {f(x):>14.6f} {df(x):>12.6f}")

    print(f"\nConverged after {iters} iterations")
    print(f"Local minimum at x = {x_min:.6f}, y = {f(x_min):.8f}")
    print("Analytical minimum: x = -3, y = 0")

    # effect of learning rate
    print("\nEffect of learning rate (starting at x = 2):")
    for a in [0.001, 0.01, 0.1, 0.5, 0.9]:
        xm, it, _ = gradient_descent(x0, a)
        print(f"  alpha = {a:<6} iterations = {it:<6} x_min = {xm:.6f}")

    xs = np.linspace(-8, 4, 200)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xs, f(xs), label="y = (x+3)^2")
    h = np.array(hist)
    plt.scatter(h[::max(1, len(h) // 30)], f(h[::max(1, len(h) // 30)]),
                c="red", s=15, label="GD steps")
    plt.scatter([x_min], [f(x_min)], c="green", s=60, label="minimum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Gradient descent path")

    plt.subplot(1, 2, 2)
    plt.plot(f(h))
    plt.xlabel("Iteration")
    plt.ylabel("y (cost)")
    plt.title("Cost vs iterations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
