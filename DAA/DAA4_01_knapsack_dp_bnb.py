"""
===============================================================================
DAA4 : Write a program to solve a 0-1 Knapsack problem using dynamic
       programming or branch and bound strategy.
       (Both methods are implemented below.)
===============================================================================

THEORY
------
1. 0/1 KNAPSACK
   n items with weight w_i and profit p_i, knapsack capacity W.
       maximise SUM p_i x_i   subject to SUM w_i x_i <= W,  x_i in {0,1}
   Items cannot be broken, so the greedy ratio method may fail.
   Brute force checks all 2^n subsets -> O(2^n). 0/1 knapsack is NP-hard
   (decision version NP-complete); DP gives a PSEUDO-POLYNOMIAL O(nW) solution.

2. DYNAMIC PROGRAMMING (Unit III)
   Principle: solve each sub-problem once, store its answer in a table and
   reuse it. Applicable when the problem has
     a) OPTIMAL SUBSTRUCTURE  (Principle of Optimality - Bellman: whatever the
        initial state and decision, the remaining decisions must be optimal
        with respect to the state resulting from the first decision)
     b) OVERLAPPING SUB-PROBLEMS.
   Approaches: top-down (memoization) or bottom-up (tabulation).
   Control abstraction:
       Algorithm DP(problem)
           characterise structure of optimal solution
           define value of optimal solution recursively
           compute values bottom-up in a table
           construct optimal solution from computed information (traceback)

3. DP RECURRENCE FOR 0/1 KNAPSACK
   K[i][w] = max profit using first i items with capacity w
       K[0][w] = 0, K[i][0] = 0
       K[i][w] = K[i-1][w]                               if w_i > w
               = max( K[i-1][w], p_i + K[i-1][w - w_i] )  otherwise
   Answer = K[n][W]. Traceback: if K[i][w] != K[i-1][w], item i is taken.
   Time O(nW), Space O(nW)  (can be reduced to O(W) with a 1-D array).

4. BRANCH AND BOUND (Unit IV)
   Explores a STATE-SPACE TREE (each level = decision include/exclude item i)
   but, unlike backtracking, uses a BOUND function to prune nodes that cannot
   lead to a better solution than the best found so far.
   Terms: live node (generated, children not yet all generated), E-node
   (node being expanded), dead node (fully expanded or pruned).
   Strategies to choose next E-node:
     - FIFO B&B  : queue (BFS)
     - LIFO B&B  : stack (DFS)
     - LC  (Least Cost) B&B : priority queue on cost/bound (best-first)
   For maximisation, the UPPER BOUND of a node = current profit + profit of
   remaining items filled greedily by ratio (fractional knapsack relaxation).
   If bound <= best profit found -> prune.
   Worst case still O(2^n) but pruning makes it much faster in practice.

5. EXAMPLE  W = 50, p = (60, 100, 120), w = (10, 20, 30)
   Greedy-by-ratio gives 160 (items 1,2) but optimal is 220 (items 2,3).

6. BACKTRACKING vs BRANCH & BOUND
   | Backtracking                      | Branch and Bound                 |
   |-----------------------------------|----------------------------------|
   | DFS of state-space tree           | BFS / LC search                  |
   | used for decision problems        | used for optimisation problems   |
   | prunes on constraint violation    | prunes on bound value too        |

CONCLUSION: DP solves 0/1 knapsack exactly in O(nW); LC Branch & Bound with a
fractional upper bound also finds the optimum while pruning the search tree.
===============================================================================
"""

import heapq


# ---------------------------------------------------------------------------
# Method 1: Dynamic Programming (bottom-up tabulation)
# ---------------------------------------------------------------------------
def knapsack_dp(profits, weights, W):
    n = len(profits)
    K = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] > w:
                K[i][w] = K[i - 1][w]
            else:
                K[i][w] = max(K[i - 1][w],
                              profits[i - 1] + K[i - 1][w - weights[i - 1]])

    # traceback to find selected items
    selected, w = [], W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.append(i)
            w -= weights[i - 1]
    return K[n][W], sorted(selected), K


# ---------------------------------------------------------------------------
# Method 2: Least-Cost Branch and Bound
# ---------------------------------------------------------------------------
def knapsack_bnb(profits, weights, W):
    n = len(profits)
    # sort by ratio so that the fractional bound is easy to compute
    order = sorted(range(n), key=lambda i: profits[i] / weights[i],
                   reverse=True)
    p = [profits[i] for i in order]
    w = [weights[i] for i in order]

    def bound(level, cur_w, cur_p):
        """upper bound: fill remaining capacity greedily (fractional)"""
        if cur_w >= W:
            return 0
        b, tw = cur_p, cur_w
        j = level
        while j < n and tw + w[j] <= W:
            tw += w[j]
            b += p[j]
            j += 1
        if j < n:
            b += (W - tw) * p[j] / w[j]
        return b

    best_profit, best_set = 0, []
    nodes_explored = 0
    # max-heap on bound: (-bound, level, weight, profit, taken_items)
    pq = [(-bound(0, 0, 0), 0, 0, 0, [])]
    while pq:
        neg_b, level, cw, cp, taken = heapq.heappop(pq)
        nodes_explored += 1
        if -neg_b <= best_profit or level == n:
            continue                                  # prune
        # child 1: include item `level`
        nw, np_ = cw + w[level], cp + p[level]
        if nw <= W:
            if np_ > best_profit:
                best_profit, best_set = np_, taken + [order[level] + 1]
            b = bound(level + 1, nw, np_)
            if b > best_profit:
                heapq.heappush(pq, (-b, level + 1, nw, np_,
                                    taken + [order[level] + 1]))
        # child 2: exclude item `level`
        b = bound(level + 1, cw, cp)
        if b > best_profit:
            heapq.heappush(pq, (-b, level + 1, cw, cp, taken))
    return best_profit, sorted(best_set), nodes_explored


def main():
    try:
        n = int(input("Enter number of items (Enter for sample data): "))
        profits = list(map(int, input("Enter profits: ").split()))
        weights = list(map(int, input("Enter weights: ").split()))
        W = int(input("Enter capacity: "))
        assert len(profits) == len(weights) == n
    except Exception:  # bad input, no input, or notebook without stdin
        profits, weights, W = [60, 100, 120], [10, 20, 30], 50
        print(f"Using sample data: profits={profits} weights={weights} W={W}")

    best, items, K = knapsack_dp(profits, weights, W)
    print("\n--- Dynamic Programming ---")
    if W <= 20:
        print("DP table K[i][w]:")
        for i, row in enumerate(K):
            print(f"i={i}:", row)
    print("Maximum profit :", best)
    print("Items selected :", items)
    print("Time O(nW), Space O(nW)")

    best_b, items_b, explored = knapsack_bnb(profits, weights, W)
    print("\n--- LC Branch and Bound ---")
    print("Maximum profit :", best_b)
    print("Items selected :", items_b)
    print("Nodes explored :", explored, f"(brute force would be {2**len(profits)} subsets)")


if __name__ == "__main__":
    main()
