"""
===============================================================================
DAA5 : Write a program to generate binomial coefficients using dynamic
       programming.
===============================================================================

THEORY
------
1. BINOMIAL COEFFICIENT
   C(n, k) = number of ways to choose k objects from n = n! / (k! (n-k)!)
   They are the coefficients in (a + b)^n = SUM_{k=0..n} C(n,k) a^(n-k) b^k.

2. RECURSIVE DEFINITION (Pascal's identity)
       C(n, 0) = C(n, n) = 1
       C(n, k) = C(n-1, k-1) + C(n-1, k)       for 0 < k < n
   Direct recursion recomputes the same values again and again
   (OVERLAPPING SUB-PROBLEMS) -> exponential time, O(2^n).

3. DYNAMIC PROGRAMMING (Unit III)
   DP = solve every sub-problem once and store the result in a table
   (tabulation, bottom-up) or cache (memoization, top-down).
   Requirements: optimal substructure + overlapping sub-problems.
   Control abstraction:
       1. characterise the structure of the solution
       2. define the value recursively
       3. compute bottom-up, storing in a table
       4. construct the answer from the table
   Time analysis of DP = (number of distinct sub-problems) x (time per
   sub-problem).

4. DP ALGORITHM (Pascal's triangle)
       Algorithm Binomial(n, k)
           for i := 0 to n do
               for j := 0 to min(i, k) do
                   if j = 0 or j = i then C[i][j] := 1
                   else C[i][j] := C[i-1][j-1] + C[i-1][j]
           return C[n][k]
   Number of additions = Theta(nk).
   Time O(nk), Space O(nk); space can be reduced to O(k) by keeping a single
   row and updating it from right to left.

5. PASCAL'S TRIANGLE (n = 5)
                 1
               1   1
             1   2   1
           1   3   3   1
         1   4   6   4   1
       1   5  10  10   5   1

6. DP vs DIVIDE & CONQUER
   Divide & conquer splits into INDEPENDENT sub-problems (merge sort);
   DP is used when sub-problems OVERLAP. Other DP examples in syllabus:
   0/1 knapsack, OBST, matrix chain multiplication.

CONCLUSION: Using a DP table (Pascal's triangle) computes C(n,k) in O(nk)
instead of exponential time for naive recursion.
===============================================================================
"""

calls = 0


def binomial_recursive(n, k):
    """naive recursion - for comparison of number of calls"""
    global calls
    calls += 1
    if k == 0 or k == n:
        return 1
    return binomial_recursive(n - 1, k - 1) + binomial_recursive(n - 1, k)


def binomial_dp(n, k):
    """bottom-up DP table, returns C(n,k) and full table"""
    C = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            if j == 0 or j == i:
                C[i][j] = 1
            else:
                C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C[n][k], C


def binomial_dp_1d(n, k):
    """space optimised O(k) version"""
    row = [0] * (k + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            row[j] += row[j - 1]
    return row[k]


def pascal_triangle(n):
    rows = []
    for i in range(n + 1):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = rows[i - 1][j - 1] + rows[i - 1][j]
        rows.append(row)
    for i, row in enumerate(rows):
        print("   " * (n - i) + "".join(f"{v:^6}" for v in row))


def main():
    global calls
    try:
        n = int(input("Enter n: "))
        k = int(input("Enter k: "))
    except Exception:  # bad input, no input, or notebook without stdin
        n, k = 6, 3
        print(f"Using sample values n={n}, k={k}")
    if not 0 <= k <= n:
        print("Require 0 <= k <= n")
        return

    value, table = binomial_dp(n, k)
    print(f"\nDP table C[i][j] (i = 0..{n}, j = 0..{k}):")
    print("i\\j " + "".join(f"{j:>6}" for j in range(k + 1)))
    for i, row in enumerate(table):
        print(f"{i:<4}" + "".join(f"{v:>6}" if j <= i else f"{'':>6}"
                                  for j, v in enumerate(row)))

    print(f"\nC({n},{k}) using DP (2-D table) = {value}")
    print(f"C({n},{k}) using DP (1-D row)   = {binomial_dp_1d(n, k)}")

    if n <= 25:
        calls = 0
        r = binomial_recursive(n, k)
        print(f"C({n},{k}) using naive recursion = {r}  "
              f"(recursive calls = {calls})")

    print(f"\nPascal's triangle up to n = {min(n, 10)}:")
    pascal_triangle(min(n, 10))
    print("\nTime complexity: DP O(nk), naive recursion O(2^n)")


if __name__ == "__main__":
    main()
