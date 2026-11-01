#!/usr/bin/env python
# coding: utf-8

# In[1]:


import heapq
import time

graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'D': 5, 'E': 6},
    'C': {'A': 2, 'D': 8, 'F': 7},
    'D': {'B': 5, 'C': 8, 'G': 3},
    'E': {'B': 6, 'G': 4},
    'F': {'C': 7, 'G': 5},
    'G': {'D': 3, 'E': 4, 'F': 5}
}

heuristic = {
    'A': 10,
    'B': 8,
    'C': 7,
    'D': 4,
    'E': 5,
    'F': 3,
    'G': 0
}


def greedy_best_first(start, goal):

    queue = [(heuristic[start], start, [start])]
    visited = set()
    expanded = 0

    while queue:

        h, node, path = heapq.heappop(queue)

        if node in visited:
            continue

        visited.add(node)
        expanded += 1

        if node == goal:
            cost = sum(graph[path[i]][path[i+1]]
                       for i in range(len(path)-1))
            return path, cost, expanded

        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(
                    queue,
                    (heuristic[neighbor], neighbor,
                     path + [neighbor])
                )

    return None, 0, expanded


def a_star(start, goal):

    queue = [(heuristic[start], 0, start, [start])]
    visited = {}
    expanded = 0

    while queue:

        f, g, node, path = heapq.heappop(queue)

        if node in visited and visited[node] <= g:
            continue

        visited[node] = g
        expanded += 1

        if node == goal:
            return path, g, expanded

        for neighbor, cost in graph[node].items():

            new_g = g + cost

            if neighbor not in visited or new_g < visited[neighbor]:

                new_f = new_g + heuristic[neighbor]

                heapq.heappush(
                    queue,
                    (new_f, new_g, neighbor,
                     path + [neighbor])
                )

    return None, 0, expanded


print("========== INFORMED SEARCH ==========")

start = input("Enter Starting Node (A-G): ").upper()
goal = input("Enter Goal Node (A-G): ").upper()

if start not in graph or goal not in graph:

    print("Invalid Node! Please enter A to G.")

else:

    t1 = time.perf_counter()
    greedy_path, greedy_cost, greedy_nodes = greedy_best_first(start, goal)
    greedy_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    astar_path, astar_cost, astar_nodes = a_star(start, goal)
    astar_time = time.perf_counter() - t2

    print("\n----- GREEDY BEST FIRST SEARCH -----")

    print("Path:", " -> ".join(greedy_path))
    print("Path Cost:", greedy_cost)
    print("Nodes Expanded:", greedy_nodes)
    print("Execution Time:", greedy_time)

    print("\n----- A* SEARCH -----")

    print("Path:", " -> ".join(astar_path))
    print("Path Cost:", astar_cost)
    print("Nodes Expanded:", astar_nodes)
    print("Execution Time:", astar_time)

    print("\n----- PERFORMANCE COMPARISON -----")

    print("Greedy Path Cost:", greedy_cost)
    print("A* Path Cost:", astar_cost)

    if greedy_cost == astar_cost:
        print("Both algorithms found paths with equal cost.")
    elif greedy_cost < astar_cost:
        print("Greedy found a lower-cost path for this example.")
    else:
        print("A* found a lower-cost path.")

    print("\nGreedy Expanded Nodes:", greedy_nodes)
    print("A* Expanded Nodes:", astar_nodes)

    print("\nExecution Completed Successfully!")


# In[ ]:




