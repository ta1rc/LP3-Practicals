"""
===============================================================================
DAA6 : Design 8-Queens matrix having first Queen placed. Use backtracking to
       place remaining Queens to generate the final 8-queen's matrix.
       (Works for general n; default n = 8.)
===============================================================================

THEORY
------
1. N-QUEENS PROBLEM
   Place n queens on an n x n chessboard so that no two queens attack each
   other, i.e. no two queens share a ROW, COLUMN or DIAGONAL.
   8-Queens has 92 solutions (12 distinct up to symmetry).

2. BACKTRACKING (Unit IV)
   Backtracking is a systematic DFS search of the STATE-SPACE TREE. The
   solution is a tuple (x1, x2, ..., xn); components are chosen one at a time.
   After each choice a BOUNDING FUNCTION (constraint check) is applied; if the
   partial solution cannot lead to a valid answer, the node is killed and the
   algorithm "backtracks" to the previous level and tries the next option.
   - Explicit constraints: each x_i in {1..n} (one queen per row)
   - Implicit constraints: no two queens in same column / diagonal
   Terms: problem state, solution state, answer state, live node, E-node,
   dead node.

3. CONTROL ABSTRACTION (recursive backtracking)
       Algorithm Backtrack(k)
       {
           for each x[k] in T(x[1..k-1]) do
               if B_k(x[1..k]) = true then          // bounding function
               {
                   if (x[1..k] is an answer) then write x[1..k]
                   else Backtrack(k + 1)
               }
       }

4. N-QUEENS USING BACKTRACKING
   x[i] = column of queen in row i.
   Place(k, i): returns true if queen k can be placed in column i:
       for j := 1 to k-1:
           if x[j] = i                         (same column)
           or |x[j] - i| = |j - k|             (same diagonal)
               return false
       return true
   Here the FIRST queen is fixed at a user-given position (row 0, column c)
   and the remaining n-1 queens are placed by backtracking.

5. TIME ANALYSIS
   Without pruning the state space has n^n (or n! with column constraint)
   nodes. Backtracking prunes most of them; worst case O(n!), Place() costs
   O(n), so worst case O(n * n!). Space O(n) for x[] + recursion stack.
   The program counts backtracks to show the pruning.

6. OTHER BACKTRACKING PROBLEMS (syllabus): graph colouring, sum of subsets,
   Hamiltonian cycle.

CONCLUSION: With the first queen fixed, backtracking efficiently finds a valid
placement for the remaining queens by abandoning partial boards that violate
constraints.
===============================================================================
"""

backtracks = 0


def is_safe(board, row, col, n):
    # check column above
    for r in range(row):
        if board[r][col] == 1:
            return False
    # upper-left diagonal
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0:
        if board[r][c] == 1:
            return False
        r, c = r - 1, c - 1
    # upper-right diagonal
    r, c = row - 1, col + 1
    while r >= 0 and c < n:
        if board[r][c] == 1:
            return False
        r, c = r - 1, c + 1
    return True


def solve(board, row, n):
    """place queens from `row` to n-1; row 0 already holds the first queen"""
    global backtracks
    if row == n:
        return True
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1          # place
            if solve(board, row + 1, n):
                return True
            board[row][col] = 0          # remove -> BACKTRACK
            backtracks += 1
    return False


def print_board(board):
    n = len(board)
    print("   " + " ".join(str(c) for c in range(n)))
    for r, row in enumerate(board):
        print(f"{r}  " + " ".join("Q" if v else "." for v in row))
    print("\nBinary matrix:")
    for row in board:
        print(row)


def main():
    global backtracks
    try:
        n = int(input("Enter board size n (Enter for 8): ") or 8)
        first_col = int(input(f"Enter column (0-{n-1}) of the first queen "
                              "in row 0: "))
    except Exception:  # bad input, no input, or notebook without stdin
        n, first_col = 8, 0
        print(f"Using n = {n}, first queen at (0, {first_col})")
    if not 0 <= first_col < n:
        print("Invalid column")
        return

    board = [[0] * n for _ in range(n)]
    board[0][first_col] = 1              # first queen pre-placed
    backtracks = 0

    if solve(board, 1, n):
        print(f"\nFinal {n}-Queens matrix (first queen at row 0, col {first_col}):")
        print_board(board)
        print("\nQueen positions (row, col):",
              [(r, row.index(1)) for r, row in enumerate(board)])
        print("Number of backtracks:", backtracks)
    else:
        print(f"No solution exists with the first queen at (0, {first_col})")


if __name__ == "__main__":
    main()
