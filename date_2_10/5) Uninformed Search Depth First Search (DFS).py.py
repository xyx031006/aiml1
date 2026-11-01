#!/usr/bin/env python
# coding: utf-8

# In[1]:


graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'G'],
    'F': ['C'],
    'G': ['E']
}

def dfs(graph, start, visited=None):

    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=" ")

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited


print("========== DEPTH FIRST SEARCH ==========")

print("\nGraph:")
for node in graph:
    print(node, ":", graph[node])

start = input("\nEnter Starting Node (A-G): ").upper()

if start in graph:
    print("\nDFS Traversal:")

    visited = dfs(graph, start)

    print("\n\nVisited Nodes:", visited)

else:
    print("Invalid Node! Please enter A to G.")

print("\nDFS Execution Completed.")


# In[ ]:




