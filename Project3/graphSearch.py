import copy
import heapq
import numpy as np
from Project3.lwnode import *

class graphSearch:
  def __init__(self, node_type):
    self.node_type = node_type
    # The heap for nodes to visit is kept here since it's a common component across different
    # search strategies
    self.heap = []

  def run(self, seed, endnode=None):
    heap = []
    start_node = self.node_type(nd=seed)
    heapq.heappush(heap, start_node)
    while heap:
      node = heapq.heappop(heap)
      if self.marked[node.nd]:
        continue
      self.mark(node)
      self.setPointer(node, node.pr)
      if node.nd == endnode:
        return self.trace(node.nd, seed), node.cost
      for neighbor in self.findNeibs(node):
        if not self.marked[neighbor.nd]:
          heapq.heappush(heap, neighbor)
    return None

  def trace(self, end_nd, seed):
    path = []
    current = end_nd
    while current != seed:
      path.append(current)
      current = self.getPointer(current)
      if current is None:  # Safety check in case of disconnected nodes
        return []
    path.append(seed)
    return path  # Reverse the path to start from seed to end_nd


class graphSearchLW(graphSearch):
  def __init__(self):
    super().__init__(lwnode)
    self.marked = None
    self.pointers = None
    self.edges = None

  def run(self, edges, seed, endnode=None):
    self.edges = edges
    self.marked = np.zeros(len(edges), dtype=np.uint8)
    self.pointers = -np.ones(len(edges), dtype=np.longlong)
    return super().run(seed, endnode)

  def isNotMarked(self, node):
    return self.marked[node.nd] == False

  def getPointer(self, nd):
    return self.pointers[nd]

  def findNeibs(self, node):
    neibs = copy.deepcopy(self.edges[node.nd])
    for n in neibs:
      n.cost += node.cost
    return neibs

  def setPointer(self, node, pr):
    self.pointers[node.nd] = pr

  def mark(self, node):
    self.marked[node.nd] = True
