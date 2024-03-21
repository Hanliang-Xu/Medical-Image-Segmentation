# % Class to perform fast marching
# % upwindEikonal function needs to be added
# % ECE 8396: Medical Image Segmentation
# % Spring 2024


import copy

import matplotlib.pyplot as plt
import numpy as np

from heap import *


class lSNode:
  def __init__(self):
    self.x = self.y = self.z = -1
    self.d = np.inf

  def __init__(self, x, y, z, d):
    self.x = x
    self.y = y
    self.z = z
    self.d = d

  def __lt__(self, rhs):
    return self.d < rhs.d


INF = 1e9


class fastMarching:
  def __init__(self, plot=False):
    self.nb = heap()
    self.nbin = []
    self.nbout = []
    self.dmap = None
    self.active = None
    self.voxsz = None
    self.plot = plot
    self.plotfreq = 25
    if plot:
      plt.ion()
      plt.pause(0.1)
    self.speed = None

  def update(self, dmap_i=None, nbdist=np.inf, voxsz=np.array([1, 1, 1]), speed=None):
    # first call needs dmap_i. Subsequent calls can update pre-existing dmap
    if dmap_i is not None:
      self.dmap = INF * np.ones(np.shape(dmap_i))
      self.voxsz = voxsz
    else:
      dmap_i = np.copy(self.dmap)
      self.dmap[:] = INF

    if speed is None:
      self.speed = np.ones(np.shape(self.dmap))
    else:
      self.speed = np.copy(speed)

    fore_mask = dmap_i <= 0
    self.active = np.copy(fore_mask)

    # build this function
    self.insertBorderVoxels(dmap_i)

    dist = 0
    nbo = []
    cnt = 0
    # loop to process heap and build foreground narrow band
    while self.nb.isEmpty() == False and dist < nbdist:
      nd = self.nb.pop()
      # flush any already finished voxels out of heap and continue
      while self.active[nd.x, nd.y, nd.z] == 0 and self.nb.isEmpty() == False:
        nd = self.nb.pop()

      dist = nd.d
      # record the new narrow band in a list for next run
      nbo.append(nd)
      # set active=0 because this voxel is finished
      self.active[nd.x, nd.y, nd.z] = 0
      # estimate distances for neighbors of this node using Eikonal equation
      self.upwindEikonal(nd)
      # iterate and plot
      cnt += 1
      if self.plot and cnt % self.plotfreq == 0:
        plt.cla()
        ax = plt.gca()
        ax.imshow(self.dmap[:, :, np.shape(self.dmap)[2] // 2], 'gray', vmin=-dist, vmax=dist)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.01)

    # all estimated distances are positive, make them negative
    self.dmap[fore_mask] *= -1

    # repeat for background
    nbo2 = []
    self.nb = heap()

    self.active = dmap_i >= 0
    self.insertBorderVoxels(dmap_i, False)
    mdist = dist
    dist = 0

    while self.nb.isEmpty() == False and dist < nbdist:
      nd = self.nb.pop()
      # flush any already finished voxels out of heap and continue
      while self.active[nd.x, nd.y, nd.z] == 0 and self.nb.isEmpty() == False:
        nd = self.nb.pop()

      dist = nd.d
      nbo2.append(nd)
      self.active[nd.x, nd.y, nd.z] = 0
      self.upwindEikonal(nd)
      cnt += 1
      if self.plot and cnt % self.plotfreq == 0:
        plt.cla()
        ax = plt.gca()
        ax.imshow(self.dmap[:, :, np.shape(self.dmap)[2] // 2], 'gray', vmin=-mdist, vmax=dist)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.01)

    self.nbin = copy.deepcopy(nbo)
    self.nbout = copy.deepcopy(nbo2)
    self.nb = heap()
    return

  def getNB(self):
    return self.nbin, self.nbout

  def insertBorderVoxels(self, dmap_i, fore=True):
    # This function estimates distances for border voxels, pushes those border
    # voxels into the heap, and sets their active status to 2.
    r, c, d = np.shape(dmap_i)
    if fore:
      direction = 1
    else:
      direction = -1
    nbo = np.concatenate((self.nbin, self.nbout), axis=0)
    plus = np.zeros(3)
    minus = np.zeros(3)

    # if first call to fast-march, narrow band does not exist yet and nbo is empty
    if len(nbo) == 0:
      initnb = 1
      # finding all x, y, z boundary voxel pairs using vectorized code
      X, Y, Z = np.meshgrid(range(r), range(c), range(d), indexing='ij')
      mskx = dmap_i[0:r - 1, :, :] * dmap_i[1:, :, :] <= 0
      msky = dmap_i[:, 0:c - 1, :] * dmap_i[:, 1:, :] <= 0
      mskz = dmap_i[:, :, 0:d - 1] * dmap_i[:, :, 1:] <= 0
      msk = np.zeros((r, c, d), dtype=bool)
      msk[0:-1, :, :] |= mskx
      msk[1:, :, :] |= mskx
      msk[:, 0:-1, :] |= msky
      msk[:, 1:, :] |= msky
      msk[:, :, 0:-1] |= mskz
      msk[:, :, 1:] |= mskz
      # boolean index into mask to flag boundary voxels
      # msk[np.concatenate((mskx, np.zeros((1,c,d), dtype=bool)),axis=0)] = True
      # msk[np.concatenate((np.zeros((1,c,d), dtype=bool),mskx),axis=0)] = True
      # msk[np.concatenate((msky,np.zeros((r,1,d),dtype=bool)),axis=1)] = True
      # msk[np.concatenate((np.zeros((r,1,d),dtype=bool),msky),axis=1)] = True
      # msk[np.concatenate((mskz,np.zeros((r,c,1),dtype=bool)),axis=2)] = True
      # msk[np.concatenate((np.zeros((r,c,1),dtype=bool),mskz),axis=2)] = True
      # obtain compact list of boundary voxel XYZ coordinates
      X = X[msk].flatten()
      Y = Y[msk].flatten()
      Z = Z[msk].flatten()
      length = np.size(X)
    else:
      # just use pre-existing narrow band
      initnb = 0
      length = len(nbo)

    for i in range(length):
      if initnb:
        x = X[i]
        y = Y[i]
        z = Z[i]
      else:
        x = nbo[i].x
        y = nbo[i].y
        z = nbo[i].z

      # only care about voxels in either the fore or background
      if dmap_i[x, y, z] * direction <= 0:
        plus[:] = minus[:] = dmap_i[x, y, z]
        # finding neighbor distances accounting for boundary
        if x < r - 1:
          plus[0] = dmap_i[x + 1, y, z]
        if x > 0:
          minus[0] = dmap_i[x - 1, y, z]
        if y < c - 1:
          plus[1] = dmap_i[x, y + 1, z]
        if y > 0:
          minus[1] = dmap_i[x, y - 1, z]
        if z < d - 1:
          plus[2] = dmap_i[x, y, z + 1]
        if z > 0:
          minus[2] = dmap_i[x, y, z - 1]

        weight = 0
        cnt = 0
        # loop over x, y, z to find build estimated distance to surface
        for j in [0, 1, 2]:
          d1 = d2 = 2 * self.voxsz[j]
          # bug was below, needs to be ge not gt
          if plus[j] * direction >= 0:  # outside desired region
            denom = (dmap_i[x, y, z] - plus[j])
            if denom == 0:
              denom = 1
            d1 = dmap_i[x, y, z] * self.voxsz[j] / denom  # always positive and<=1

          if minus[j] * direction >= 0:  # outside desired region
            denom = (dmap_i[x, y, z] - minus[j])
            if denom == 0:
              denom = 1
            d2 = dmap_i[x, y, z] * self.voxsz[j] / denom

          ndist = d1 if d1 < d2 else d2
          if ndist == 0:
            cnt = -1
            break

          if ndist < 2 * self.voxsz[j]:
            cnt += 1
            weight += 1 / (ndist * ndist)
        # if at least one valid distance is found add to the narrow band heap and set active=2

        if cnt < 0:
          self.dmap[x, y, z] = 0
          self.nb.insert(lSNode(x, y, z, 0))
          self.active[x, y, z] = 2
        if cnt > 0:
          ndist = 1 / np.sqrt(weight) / self.speed[x, y, z]
          self.dmap[x, y, z] = ndist
          self.nb.insert(lSNode(x, y, z, ndist))
          self.active[x, y, z] = 2

  def upwindEikonal(self, node):
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    r, c, d = self.dmap.shape
    for dx, dy, dz in neighbors:
      x, y, z = node.x + dx, node.y + dy, node.z + dz
      if 0 <= x < r and 0 <= y < c and 0 <= z < d and self.active[x, y, z] == 1:
        # Compute tentative distances based on the Eikonal equation
        a_dist, b_dist, c_dist = np.inf, np.inf, np.inf
        if 0 <= x - 1 < r:
          a_dist = self.dmap[x - 1, y, z]
        if x + 1 < r:
          a_dist = min(a_dist, self.dmap[x + 1, y, z])
        if 0 <= y - 1 < c:
          b_dist = self.dmap[x, y - 1, z]
        if y + 1 < c:
          b_dist = min(b_dist, self.dmap[x, y + 1, z])
        if 0 <= z - 1 < d:
          c_dist = self.dmap[x, y, z - 1]
        if z + 1 < d:
          c_dist = min(c_dist, self.dmap[x, y, z + 1])

        # Ensure a, b, c are sorted such that a <= b <= c
        a_dist, b_dist, c_dist = sorted([a_dist, b_dist, c_dist])

        # Calculate new distance using the Eikonal equation
        F = 1.0 / self.speed[x, y, z]
        distance = self.solveEikonal(a_dist, b_dist, c_dist, F)

        # Update the distance map and active set if the new distance is smaller
        if distance < self.dmap[x, y, z]:
          self.dmap[x, y, z] = distance
          self.nb.insert(lSNode(x, y, z, distance))  # Add node to heap

  def solveEikonal(self, a, b, c, F):
    """
    Solves the Eikonal equation given three smallest distances from neighbors (a, b, c)
    and the local speed (F) at the node. Returns the updated distance T to the node.

    Parameters:
    - a, b, c: float, the three smallest neighbor distances, where a <= b <= c
    - F: float, the local speed at the node

    Returns:
    - T: float, the updated distance to the node
    """

    # Calculate the argument of the square root to check if it's non-negative
    sqrt_arg = 2 * F ** 2 - (a - b) ** 2

    if sqrt_arg >= 0:
      # Safe to take the square root
      T = (a + b + np.sqrt(sqrt_arg)) / 2
    else:
      # Fallback to using only the smallest distance and F if the square root argument is negative
      T = a + F

    # Further check to ensure T does not exceed c, indicating an issue with the chosen distances
    if T > c:
      T = a + F

    return T
