from collections import defaultdict


# ------------------------ Topological Sort ------------------------ 
"""
Topological Sort is used to find a linear ordering of vertices in a 
Directed Acyclic Graph (DAG) such that 
for every directed edge uv from vertex u to vertex v, u comes before v in the ordering.
This uses DFS to find the topological ordering of the vertices.
"""
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vertices = vertices
    
    def add_edge(self, u, v):
        """Add an edge from vertex u to vertex v"""
        self.graph[u].append(v)
    
    def topological_sort_util(self, v, visited, stack):
        """Recursive utility function used by topological_sort"""
        # Mark the current node as visited
        visited[v] = True
        
        # Recur for all adjacent vertices
        for adjacent in self.graph[v]:
            if not visited[adjacent]:
                self.topological_sort_util(adjacent, visited, stack)
        
        # Push current vertex to stack which stores the result
        stack.append(v)
    
    def topological_sort(self):
        """
        Perform topological sort on the graph
        Returns a list of vertices in topological order
        """
        # Mark all vertices as not visited
        visited = [False] * self.vertices
        stack = []
        
        # Call the recursive helper function for all vertices
        for i in range(self.vertices):
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)
        
        # Return the stack in reverse order
        return stack[::-1]

# ------------------------ Prefix Tree ------------------------ 
"""
A prefix tree, also known as a trie, is a tree data structure used to store a dynamic set of strings.
It is used to efficiently store and retrieve strings with common prefixes.
It uses a hashmap to store the children of each node.
"""
class Trie:

    def __init__(self):
        self.children = {}
        self.end = False

    def insert(self, word: str) -> None:
        for w in word:
            # create a new children
            if w not in self.children:
                self.children[w] = Trie()
            self = self.children[w]
            
        self.end = True

    def search(self, word: str) -> bool:
        for w in word:
            if w not in self.children: return False
            self = self.children[w]
        return self.end

    def startsWith(self, prefix: str) -> bool:
        for w in prefix:
            if w not in self.children: return False
            self = self.children[w]
        return True


