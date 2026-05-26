# Appendix: Algorithm Invocation Steps (n=10)

Source: `ALGO_P2_phase2_normal.csv` (`reasoning_type = algorithm_invocation`).
Phase-1 declarations from `ALGO_P2_phase1_*_new.csv`.

| Case | Problem | Subtype | Instance | Model | Step | Correct | Phase-1 declared | Phase-2 invocation (abridged) |
|------|---------|---------|----------|-------|------|---------|------------------|-------------------------------|
| A1 | SP_002 | shortest_path | standard | llama-3.1-8b-instruct | 4 | No | Dijkstra | I chose to move to node 5 because it has the lowest edge weight among the unvisited neighbors of the… |
| A2 | SP_004 | shortest_path | adversarial | claude-sonnet-4 | 0 | No | Dijkstra | Node 1 has the lowest cost (2) among all unvisited neighbors, following the greedy approach of Dijks… |
| A3 | SP_037 | shortest_path | adversarial | claude-sonnet-4 | 0 | No | Dijkstra's algorithm | Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of D… |
| A4 | SP_040 | shortest_path | adversarial | claude-sonnet-4 | 0 | No | Dijkstra's algorithm | Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of D… |
| A5 | SP_044 | shortest_path | adversarial | llama-3.1-8b-instruct | 2 | No | Dijkstra | I chose this node because it is the only unvisited neighbor of the current node 3, and according to … |
| A6 | SP_062 | shortest_path | adversarial | claude-sonnet-4 | 0 | No | Dijkstra's algorithm | Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of D… |
| A7 | SP_065 | shortest_path | adversarial | claude-sonnet-4 | 0 | No | Dijkstra | Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of D… |
| A8 | WIS_003 | wis | adversarial | llama-3.1-8b-instruct | 1 | No | Dynamic Programming | The first interval has the highest weight among all available intervals, making it the most profitab… |
| A9 | WIS_007 | wis | standard | llama-3.1-8b-instruct | 2 | No | the Knapsack algorithm with a variation to solve this weighted interval scheduling problem | Interval 4 has the highest weight (7) among the available intervals, so it is the most profitable to… |
| A10 | WIS_020 | wis | adversarial | claude-sonnet-4 | 0 | No | dynamic programming | Interval 4 has the highest weight (30) among all available intervals, and selecting it first allows … |

---

## Full case entries

### A1: SP_002 — llama-3.1-8b-instruct

- **Subtype / instance:** shortest_path / standard
- **Step index:** 4
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra
- **Phase-1 greedy assessment:** No — a greedy approach will not work here. This is because the locally optimal choice (the shortest neighbor) is not guaranteed to lead to a globally optimal path, due to the existence of indirect paths and multiple routes between locations.

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm to solve the shortest path problem. Dijkstra's algorithm is a suitable choice for this problem because it can handle graphs with non-negative edge weights and is guaranteed to find the shortest path to the destination. The given trail network has many non-negative travel times, and using Dijkstra's algorithm will help ensure we find the optimal solution.


**Phase-2 step text (full):**

> I chose to move to node 5 because it has the lowest edge weight among the unvisited neighbors of the current node, which is a heuristic commonly used in shortest path algorithms like Dijkstra's or A*.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A2: SP_004 — claude-sonnet-4

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra
- **Phase-1 greedy assessment:** No — a greedy approach will not work here because the time-dependent nature of edges means that taking a seemingly faster immediate route could result in arriving at a node at an unfavorable time, making subsequent edges much more expensive and leading to

**Phase-1 declaration (opening):**

> 1. I will use a modified Dijkstra's algorithm that accounts for time-dependent edge costs. Since the edge costs depend on the arrival time at nodes, I need to track both the current node and the current time as part of the state. Traditional Dijkstra's won't work because the cost of an edge changes based on when you traverse it, so I need to consider time as part of the state space and potentially explore multiple arrival times at the same node.


**Phase-2 step text (full):**

> Node 1 has the lowest cost (2) among all unvisited neighbors, following the greedy approach of Dijkstra's algorithm to always choose the minimum cost edge first.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A3: SP_037 — claude-sonnet-4

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra's algorithm
- **Phase-1 greedy assessment:** No — a greedy approach will not work here because shortest path problems often require considering longer intermediate paths that lead to globally optimal solutions, and the greedy strategy of always choosing the nearest unvisited neighbor can miss these 

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm because this is a shortest path problem on a weighted directed graph with non-negative edge weights, and Dijkstra's algorithm is optimal for finding the shortest path from a single source to all other nodes in such graphs.


**Phase-2 step text (full):**

> Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of Dijkstra's algorithm, we should explore the cheapest available path first.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A4: SP_040 — claude-sonnet-4

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra's algorithm
- **Phase-1 greedy assessment:** No — a greedy approach of always moving to the nearest unvisited neighbor will not work here because shortest path problems often require considering longer intermediate steps that lead to globally optimal solutions, and the greedy choice of nearest neigh

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm because this is a shortest path problem in a weighted directed graph with non-negative edge weights, and Dijkstra's algorithm is optimal for finding the shortest path from a single source to all other vertices in such graphs. It systematically explores paths in order of increasing cost, guaranteeing that when a node is finalized, the shortest path to it has been found.


**Phase-2 step text (full):**

> Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of Dijkstra's algorithm, we should explore the nearest unvisited node first.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A5: SP_044 — llama-3.1-8b-instruct

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 2
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra
- **Phase-1 greedy assessment:** No — a greedy approach will not work here. This is because a greedy approach would always choose the locally shortest edge, which does not guarantee finding the globally shortest path in a graph with negative weights or cycles. In this graph, we have nega

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm to solve this problem. Dijkstra's algorithm is a suitable choice for this problem because it finds the shortest path between two nodes in a weighted graph, and it can handle negative or zero weights. Additionally, Dijkstra's algorithm is efficient for this problem, with a time complexity of O(|E|log|V|) when implemented with a binary heap.


**Phase-2 step text (full):**

> I chose this node because it is the only unvisited neighbor of the current node 3, and according to Dijkstra's algorithm, we should choose the shortest path, which in this case is directly to node 6 with a cost of 1.


**Declaration–execution gap:** 
Names Dijkstra; execution unclear.


---

### A6: SP_062 — claude-sonnet-4

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra's algorithm
- **Phase-1 greedy assessment:** No — a greedy approach of always moving to the nearest unvisited neighbor will not work here because it makes locally optimal choices without considering the global picture, which can lead to suboptimal paths when a longer initial edge leads to a much sho

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm because this is a shortest path problem in a weighted directed graph with non-negative edge weights, and Dijkstra's algorithm guarantees finding the optimal shortest path from a single source to all other vertices by systematically exploring paths in order of increasing total cost.


**Phase-2 step text (full):**

> Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of Dijkstra's algorithm, we should explore the nearest unvisited node first.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A7: SP_065 — claude-sonnet-4

- **Subtype / instance:** shortest_path / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dijkstra
- **Phase-1 greedy assessment:** No — a greedy approach of always moving to the nearest unvisited neighbor will not work here because it makes locally optimal choices without considering the global picture, potentially missing shorter paths that initially involve taking longer edges but 

**Phase-1 declaration (opening):**

> 1. I will use Dijkstra's algorithm because this is a shortest path problem in a weighted directed graph with non-negative edge weights, and Dijkstra's algorithm guarantees finding the optimal shortest path from a single source to all other vertices by systematically exploring paths in order of increasing total cost.


**Phase-2 step text (full):**

> Node 1 has the lowest cost (1) among all unvisited neighbors, and following the greedy approach of Dijkstra's algorithm, I should always move to the neighbor with the smallest edge weight first.


**Declaration–execution gap:** 
Names Dijkstra but step reduces to greedy nearest-neighbor edge choice (no priority queue / relaxation evidence).


---

### A8: WIS_003 — llama-3.1-8b-instruct

- **Subtype / instance:** wis / adversarial
- **Step index:** 1
- **Final answer correct:** No
- **Phase-1 declared algorithm:** Dynamic Programming
- **Phase-1 greedy assessment:** No — a greedy approach will not work here because it would prioritize selecting the highest-weight available interval without considering the potential benefits of selecting a lower-weight interval that does not overlap with previously selected intervals.

**Phase-1 declaration (opening):**

> 1. I will use the Dynamic Programming algorithm to solve this problem. This algorithm is suitable for this problem because it can handle overlapping intervals and compute the maximum total weight of non-overlapping intervals.


**Phase-2 step text (full):**

> The first interval has the highest weight among all available intervals, making it the most profitable choice.


**Declaration–execution gap:** 
Declares DP/optimal scheduling but executes a greedy weight-maximization step.


---

### A9: WIS_007 — llama-3.1-8b-instruct

- **Subtype / instance:** wis / standard
- **Step index:** 2
- **Final answer correct:** No
- **Phase-1 declared algorithm:** the Knapsack algorithm with a variation to solve this weighted interval scheduling problem
- **Phase-1 greedy assessment:** No — a greedy approach will not work here. The reason is that a greedy approach would always pick the highest-weight available interval, but it may not always select the optimal set of non-overlapping intervals, as it does not consider the long-term conse

**Phase-1 declaration (opening):**

> 1. I will use the Knapsack algorithm with a variation to solve this weighted interval scheduling problem. This algorithm is suitable because it allows for the selection of a subset of items (intervals) with a maximum size constraint (the timeline), where each item has a weight and a value (weight), to maximize the total value (total weight).


**Phase-2 step text (full):**

> Interval 4 has the highest weight (7) among the available intervals, so it is the most profitable to select it.


**Declaration–execution gap:** 
See texts above.


---

### A10: WIS_020 — claude-sonnet-4

- **Subtype / instance:** wis / adversarial
- **Step index:** 0
- **Final answer correct:** No
- **Phase-1 declared algorithm:** dynamic programming
- **Phase-1 greedy assessment:** No — a greedy approach will not work here because selecting the highest-weight interval early might block multiple other intervals that together have a higher total weight than the single high-weight interval.

**Phase-1 declaration (opening):**

> 1. I will use dynamic programming to solve this weighted interval scheduling problem. Dynamic programming is necessary here because we need to consider all possible combinations of non-overlapping intervals to find the globally optimal solution, and greedy approaches can lead to suboptimal results when higher-weight intervals conflict with multiple smaller intervals that together have greater total weight.


**Phase-2 step text (full):**

> Interval 4 has the highest weight (30) among all available intervals, and selecting it first allows us to potentially build a high-value solution around this most profitable interval.


**Declaration–execution gap:** 
Declares DP/optimal scheduling but executes a greedy weight-maximization step.


---
