"""
===============================================================================
DAA2 : Implement job sequencing with deadlines using a greedy method.
===============================================================================

THEORY
------
1. GREEDY STRATEGY (Unit III)
   A greedy algorithm builds a solution step by step, at each step choosing
   the option that looks best at that moment (locally optimal choice), hoping
   that this leads to a globally optimal solution. Once a choice is made it is
   never reconsidered (no backtracking).
   A problem can be solved greedily if it has:
     a) GREEDY-CHOICE PROPERTY : a global optimum can be reached by making a
        locally optimal choice.
     b) OPTIMAL SUBSTRUCTURE   : an optimal solution contains optimal solutions
        to its sub-problems.
   Terms: FEASIBLE solution = satisfies constraints; OPTIMAL solution =
   feasible solution that maximises / minimises the objective function.

2. CONTROL ABSTRACTION OF GREEDY
       Algorithm Greedy(a, n)
       {
           solution := {}
           for i := 1 to n do
           {
               x := Select(a)              // pick best remaining candidate
               if Feasible(solution U x) then
                   solution := Union(solution, x)
           }
           return solution
       }
   Time analysis: if Select costs O(1) after sorting, total = O(n log n) for
   sorting + n * (cost of Feasible).

3. JOB SEQUENCING WITH DEADLINES - PROBLEM
   Given n jobs, each with deadline d_i and profit p_i. Each job takes 1 unit
   of time; only one machine; profit p_i is earned only if job i finishes on or
   before d_i. Find a subset (and order) of jobs that maximises total profit.

4. GREEDY APPROACH
   - Sort jobs in NON-INCREASING order of profit.
   - For each job, place it in the LATEST free time slot <= its deadline
     (keeps earlier slots open for jobs with tighter deadlines).
   - If no free slot exists, reject the job.

   Algorithm JobSequencing(d, p, n)
       sort jobs by p descending
       slot[1..maxDeadline] := empty
       for each job i in sorted order:
           for t := min(maxD, d_i) down to 1:
               if slot[t] is empty: slot[t] := i ; break

5. COMPLEXITY
   Sorting: O(n log n).  Slot search: O(n) per job in the worst case.
   Overall worst-case time: O(n^2);  Space: O(n).
   (Using Disjoint-Set Union to find the latest free slot reduces it to
   O(n log n).)

6. EXAMPLE
   Jobs : J1  J2  J3  J4  J5
   Prof : 100 19  27  25  15
   Dead :  2   1   2   1   3
   Sorted by profit: J1(100), J3(27), J4(25), J2(19), J5(15)
   J1 -> slot 2, J3 -> slot 1, J4 rejected, J2 rejected, J5 -> slot 3
   Sequence: J3 -> J1 -> J5 ; Max profit = 142

7. RELATED GREEDY PROBLEMS: fractional knapsack, activity selection,
   Huffman coding, Prim's / Kruskal's MST (case study: "rail tracks connecting
   all the cities" is a minimum spanning tree problem).

CONCLUSION: Greedy selection by highest profit with latest-slot placement
gives the optimal schedule for unit-time jobs with deadlines.
===============================================================================
"""


def job_sequencing(jobs):
    """jobs: list of tuples (job_id, deadline, profit)"""
    # Step 1: sort by profit descending (greedy choice)
    jobs = sorted(jobs, key=lambda j: j[2], reverse=True)
    max_deadline = max(j[1] for j in jobs)

    slots = [None] * (max_deadline + 1)     # index 1..max_deadline used
    total_profit = 0

    print("\nStep-by-step greedy selection:")
    for job_id, deadline, profit in jobs:
        placed = False
        # Step 2: find latest free slot on or before deadline (feasibility)
        for t in range(min(deadline, max_deadline), 0, -1):
            if slots[t] is None:
                slots[t] = job_id
                total_profit += profit
                placed = True
                print(f"  {job_id} (profit={profit}, deadline={deadline}) "
                      f"-> placed in slot {t}")
                break
        if not placed:
            print(f"  {job_id} (profit={profit}, deadline={deadline}) "
                  f"-> rejected (no free slot)")

    sequence = [s for s in slots[1:] if s is not None]
    return sequence, total_profit


def read_jobs():
    try:
        n = int(input("Enter number of jobs (press Enter for sample data): "))
    except Exception:  # bad input, no input, or notebook without stdin
        return None
    jobs = []
    for i in range(n):
        d, p = map(int, input(f"Job J{i+1} -> deadline profit: ").split())
        jobs.append((f"J{i+1}", d, p))
    return jobs


def main():
    jobs = read_jobs()
    if not jobs:
        jobs = [("J1", 2, 100), ("J2", 1, 19), ("J3", 2, 27),
                ("J4", 1, 25), ("J5", 3, 15)]
        print("Using sample data")

    print(f"\n{'Job':<5}{'Deadline':>10}{'Profit':>10}")
    for j in jobs:
        print(f"{j[0]:<5}{j[1]:>10}{j[2]:>10}")

    sequence, profit = job_sequencing(jobs)
    print("\nOptimal job sequence :", " -> ".join(sequence))
    print("Maximum profit       :", profit)
    print("Time complexity      : O(n^2) (O(n log n) sorting + slot search)")


if __name__ == "__main__":
    main()
