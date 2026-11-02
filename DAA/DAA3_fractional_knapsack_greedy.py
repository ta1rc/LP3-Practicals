"""
===============================================================================
DAA3 : Write a program to solve a fractional Knapsack problem using a greedy
       method.
===============================================================================

THEORY
------
1. KNAPSACK PROBLEM
   Given n items, item i has weight w_i and profit (value) p_i, and a knapsack
   of capacity M. Choose x_i (fraction of item i taken) to
        maximise   SUM p_i * x_i
        subject to SUM w_i * x_i <= M,   0 <= x_i <= 1
   - FRACTIONAL knapsack : x_i may be any real in [0, 1]  -> GREEDY works.
   - 0/1 knapsack        : x_i in {0, 1}                  -> needs DP / B&B.

2. GREEDY STRATEGY (Unit III) - see DAA2 for principle & control abstraction.
   Three possible greedy criteria:
     a) maximum profit first        -> not always optimal
     b) minimum weight first        -> not always optimal
     c) maximum profit/weight ratio -> ALWAYS OPTIMAL for fractional knapsack
   Proof idea (exchange argument): if an optimal solution takes less of a
   higher-ratio item and more of a lower-ratio item, swapping equal weight
   increases (or keeps) profit, so the ratio-greedy solution is optimal.

3. ALGORITHM GreedyKnapsack(M, n)
       compute r_i = p_i / w_i and sort items so r_1 >= r_2 >= ... >= r_n
       x[1..n] := 0 ; U := M
       for i := 1 to n:
           if w_i > U: break
           x_i := 1 ; U := U - w_i
       if i <= n: x_i := U / w_i      // take fraction of next item

4. COMPLEXITY
   Sorting by ratio: O(n log n); filling: O(n).
   Total time O(n log n), space O(n).

5. EXAMPLE  (M = 50)
   Item  : 1    2    3
   Profit: 60   100  120
   Weight: 10   20   30
   Ratio : 6    5    4
   Take item1 fully (10), item2 fully (20), 20/30 of item3
   Profit = 60 + 100 + 120*(2/3) = 240

6. GREEDY vs DYNAMIC PROGRAMMING
   | Greedy                          | Dynamic Programming              |
   |---------------------------------|----------------------------------|
   | one decision sequence           | many decision sequences          |
   | locally optimal choice          | combines optimal sub-solutions   |
   | faster, less memory             | slower, uses table               |
   | not always optimal              | always optimal (if applicable)   |
   | e.g. fractional knapsack        | e.g. 0/1 knapsack                |

CONCLUSION: Sorting by profit/weight ratio and filling greedily yields the
optimal solution to the fractional knapsack problem in O(n log n).
===============================================================================
"""


def fractional_knapsack(profits, weights, capacity):
    n = len(profits)
    items = sorted(range(n), key=lambda i: profits[i] / weights[i],
                   reverse=True)
    x = [0.0] * n
    remaining = capacity
    total = 0.0

    print("\nItems sorted by profit/weight ratio:")
    print(f"{'Item':<6}{'Profit':>8}{'Weight':>8}{'Ratio':>8}")
    for i in items:
        print(f"{i+1:<6}{profits[i]:>8}{weights[i]:>8}"
              f"{profits[i]/weights[i]:>8.2f}")

    for i in items:
        if remaining <= 0:
            break
        if weights[i] <= remaining:          # take whole item
            x[i] = 1.0
            remaining -= weights[i]
            total += profits[i]
        else:                                # take fraction and stop
            x[i] = remaining / weights[i]
            total += profits[i] * x[i]
            remaining = 0
    return total, x


def main():
    try:
        n = int(input("Enter number of items (Enter for sample data): "))
        profits = list(map(float, input("Enter profits: ").split()))
        weights = list(map(float, input("Enter weights: ").split()))
        capacity = float(input("Enter knapsack capacity: "))
        assert len(profits) == len(weights) == n
    except Exception:  # bad input, no input, or notebook without stdin
        profits, weights, capacity = [60, 100, 120], [10, 20, 30], 50
        print("Using sample data: profits", profits, "weights", weights,
              "capacity", capacity)

    total, x = fractional_knapsack(profits, weights, capacity)
    print("\nFraction of each item taken (x_i):")
    for i, xi in enumerate(x):
        print(f"  Item {i+1}: {xi:.3f}")
    print(f"\nMaximum profit = {total:.2f}")
    print("Time complexity = O(n log n)")


if __name__ == "__main__":
    main()
