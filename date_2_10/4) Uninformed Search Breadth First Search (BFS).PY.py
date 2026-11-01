#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

def bfs(graph, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:

        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append(new_path)

    return None


print("========== BREADTH FIRST SEARCH ==========")

print("\nGraph:")
for node in graph:
    print(node, ":", graph[node])

start = input("\nEnter Starting Node (A-G): ").upper()
goal = input("Enter Goal Node (A-G): ").upper()

if start not in graph or goal not in graph:
    print("Invalid Node! Please enter A to G.")

else:
    result = bfs(graph, start, goal)

    if result:
        print("\nShortest Path:", " -> ".join(result))
        print("Path Length:", len(result) - 1)
    else:
        print("\nGoal Node Not Found.")

print("\nBFS Execution Completed.")


# In[ ]:




