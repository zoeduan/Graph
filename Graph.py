"""
Acyclic Graph or Directed Acyclic Graph
Yuehua Duan
"""

import numpy as np
import networkx as nx
from collections import deque
from collections import defaultdict

# Find the topological order of vertices
def topoSort(graph):
    # Determine all the in-degree for nodes
    in_degree = { u : 0 for u in graph } 
    for u in graph:             
        for v in graph[u]:
            in_degree[v] += 1
    # Store the Zero in-degree into Q
    Q = deque() 
    for u in in_degree:
        if in_degree[u] == 0:
            Q.appendleft(u)
    # Use L to store the topological order of vertices
    L = []

    while Q:
        u = Q.pop() 
        L.append(u) 
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                Q.appendleft(v)

    if len(L) == len(graph):
        return L
    else: 
        return []


# To determine whether graph is a directed acyclic graph(dag) or not
def is_directed_acyclic_graph(G):
    List = topoSort(G)
    if List == []:
        return False
    else:
        return True
        

# Find the cycles
# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.cycles.simple_cycles.html
def simple_cycles(G):
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()

    sccs = list(nx.strongly_connected_components(G))
    while sccs:
        scc=sccs.pop()

        startnode = scc.pop()

        path=[startnode]
        blocked = set() 
        closed = set() 
        blocked.add(startnode)
        B=defaultdict(set) 
        stack=[ (startnode,list(G[startnode])) ] 
        while stack:
            thisnode,nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()

                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)

                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append( (nextnode,list(G[nextnode])) )
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue

            if not nbrs:  
                if thisnode in closed:
                    _unblock(thisnode,blocked,B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()

                path.pop()

        G.remove_node(startnode)
        H = G.subgraph(scc)  
        sccs.extend(list(nx.strongly_connected_components(H)))


def strongly_connected_components(graph):

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []
    
    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
    
        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node],lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node],index[successor])

        if lowlink[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    
    for node in graph:
        if node not in index:
            _strong_connect(node)
    
    return result

def remove_node(G, target):
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)

def subgraph(G, vertices):
    return {v: G[v] & vertices for v in vertices}


def matrix2graph(fileName): 
    # Read n
    with open(fileName, 'r') as f:
        first_line = f.readline()
        print("Read a Graph Matrix with dimension", first_line)
	# Read an adjacency matrix of a directed graph with n vertices
    adjacenyMatrix = np.genfromtxt(fileName, skip_header=2)
    rows, cols = np.where(adjacenyMatrix == 1)
    edges = zip(rows.tolist(),cols.tolist())
    directGraph = nx.DiGraph()
    directGraph.add_edges_from(edges)
    

    if (is_directed_acyclic_graph(directGraph) == True):
        topoOrder = topoSort(directGraph)
        print('This is a Directed Acyclic Graph, Please refer the topological order of the vertices')
        for task in topoOrder:
            print(task)
        print('\n\n')
    else:
        cycles = simple_cycles(directGraph)
        print('This is a Directed Cycle Graph, please refer the direct cycle(s)')
        for cycle in cycles:
            print(cycle)
        print('\n\n')


files = ['/users/zoe/desktop/Testcase1.txt', '/users/zoe/desktop/Testcase2.txt', '/users/zoe/desktop/Testcase3.txt', '/users/zoe/desktop/Testcase4.txt' ]
for file in files:
    matrix2graph(file)
