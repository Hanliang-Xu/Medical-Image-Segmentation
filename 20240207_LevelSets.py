from heap import *
import numpy as np
import matplotlib.pyplot as plt
import copy

# class for heap-sorting voxels for distance estimation while fast-marching
class lSNode:
    def __init__(self, x=-1, y=-1, z=-1, d=np.inf):
        self.x = x
        self.y = y
        self.z = z
        self.d = d

    def __lt__(self, rhs): # less-than operator
        return self.d < rhs.d

from fastMarching import *
class fastMarchingDemo(fastMarching): # imports the upwindEikonal function that will be part of project
    def __init__(self):
        self.dmap = None
        self.active=None# 0, 1, or 2 for inactive/finished, still estimating, boundary voxels
        self.nb = heap()
        self.voxsz = np.array([1., 1., 1.])

    def update(self, dmap_i=None, nbdist=np.inf, voxsz=np.array([1.,1.,1.])):
        if dmap_i is not None:
            self.dmap = np.inf*np.ones(np.shape(dmap_i))
            self.voxsz = voxsz
        else:
            dmap_i = np.copy(self.dmap)
            self.dmap[:] = np.inf

        fore_mask = dmap_i<=0# foreground voxels active
        self.active = np.copy(fore_mask)

        # insert voxels adjacent to segmentation boundary into heap
        self.insertBorderVoxels(dmap_i)

        dist = 0
        # estimate distances for all voxels in narrow band starting with closest first
        while self.nb.isEmpty()==False and dist < nbdist:
            nd = self.nb.pop()
            # flush any nodes already finalized with lower distance in previous iterations
            while self.active[nd.x, nd.y, nd.z] == 0 and self.nb.isEmpty()==False:
                nd = self.nb.pop()
            if self.active[nd.x, nd.y, nd.z] == 0:
                break

            dist = nd.d
            self.active[nd.x, nd.y, nd.z] = 0 # popped from heap-> so estimated distance is final
            self.upwindEikonal(nd)

        self.dmap[fore_mask] *= -1# convert foreground distances to negative

        self.nb = heap()
        self.active = dmap_i >= 0 # background voxels active
        self.insertBorderVoxels(dmap_i, foreground=False)

        #copy-pasted while loop from above
        dist = 0
        while self.nb.isEmpty() == False and dist<nbdist:
            nd = self.nb.pop()
            # flush any nodes already finalized with lower distance in previous iterations
            while self.active[nd.x, nd.y, nd.z] == 0 and self.nb.isEmpty()==False:
                nd = self.nb.pop()
            if self.active[nd.x, nd.y, nd.z] == 0:
                break

            dist = nd.d
            self.active[nd.x, nd.y, nd.z] = 0
            self.upwindEikonal(nd)

    def insertBorderVoxels(self, dmap_i, foreground=True):
        # insert voxels adjacent to segmentation boundary into heap, set active=2
        if foreground:
            direction = 1
        else:
            direction = -1
        r,c,d = np.shape(dmap_i)
        plus = np.zeros(3)
        minus=np.zeros(3)

        # vectorized code to find voxels adjacent to segmentation boundary
        X,Y,Z = np.meshgrid(range(r), range(c), range(d), indexing='ij')
        mskx = dmap_i[0:-1,:,:] * dmap_i[1:,:,:] <=0
        msky = dmap_i[:,0:-1,:] * dmap_i[:,1:,:] <=0
        mskz = dmap_i[:,:,0:-1] * dmap_i[:,:,1:] <=0
        msk = np.zeros((r,c,d), dtype=bool)
        msk[0:-1,:,:] |= mskx
        msk[1:  ,:,:] |= mskx
        msk[:,0:-1,:] |= msky
        msk[:,1:  ,:] |= msky
        msk[:,:,0:-1] |= mskz
        msk[:,:,1:  ] |= mskz

        X = X[msk].flatten()
        Y = Y[msk].flatten()
        Z = Z[msk].flatten()

        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            z = Z[i]

            if dmap_i[x,y,z] == 0:
                self.dmap[x,y,z] = 0
                self.active[x,y,z] = 2
                self.nb.insert(lSNode(x, y, z, 0))

            if dmap_i[x,y,z]*direction < 0:
                plus[:]=minus[:] = dmap_i[x,y,z]
                if x<r-1:
                    plus[0] = dmap_i[x+1,y,z]
                if x>0:
                    minus[0] = dmap_i[x-1,y,z]
                if y<c-1:
                    plus[1] = dmap_i[x,y+1,z]
                if y>0:
                    minus[1] = dmap_i[x,y-1,z]
                if z<d-1:
                    plus[2] = dmap_i[x,y,z+1]
                if z>0:
                    minus[2] = dmap_i[x,y,z-1]

                weight = 0
                cnt = 0
                for j in [0,1,2]:
                    d1 = d2 = 2.0 * self.voxsz[j]
                    if plus[j]*direction >=0: # this neighbor is across boundary
                        d1 = dmap_i[x,y,z] * self.voxsz[j] / (dmap_i[x,y,z] - plus[j])
                    if minus[j]*direction >=0: # this neighbor is across boundary
                        d2 = dmap_i[x,y,z] * self.voxsz[j] / (dmap_i[x,y,z] - minus[j])

                    #closest distance in plus/minus j direction
                    ndist = d1 if d1<d2 else d2

                    if ndist < 2 * self.voxsz[j]:
                        cnt += 1
                        weight += 1/(ndist*ndist)

                # if at least 1 valid distance, add to heap with active=2
                if cnt > 0:
                    ndist = 1/np.sqrt(weight)
                    self.dmap[x,y,z] = ndist
                    self.nb.insert(lSNode(x,y,z, ndist))
                    self.active[x,y,z] = 2

# test with circle of radius=10 (have ground truth Euclidean distance map)
X, Y = np.meshgrid(range(60), range(50), indexing='ij')
img = np.sqrt((X - 29)*(X-29) + (Y-24)*(Y-24)) - 10

fig, ax = plt.subplots()
fm = fastMarchingDemo()
# fm.update(img[:,:,np.newaxis])
# ax.imshow(fm.dmap[:,:,0].T, 'gray', vmin=0, vmax=2)
# ax.imshow(fm.dmap[:,:,0].T, 'gray', vmin=0, vmax=10)
# ax.imshow(fm.dmap[:,:,0].T, 'gray', vmin=-10, vmax=2)
# cols = ax.imshow(fm.dmap[:,:,0].T, 'gray')
fm.update(img[:,:,np.newaxis], nbdist=3.5, voxsz=np.array([1.0, 0.5, 1.0]))
cols = ax.imshow(fm.dmap[:,:,0].T, 'gray', vmin=-3.5, vmax=3.5)
plt.contour(X,Y, img, [0.0], colors=[[1.,0.,0.]])
plt.colorbar(cols)
ax.set_aspect(0.5)
plt.show()

# test error
# errorImg = np.abs(fm.dmap[:,:,0] - img)
# print(f'Mean Abs Error: {np.mean(errorImg)}')
# fig, ax = plt.subplots()
# cols = ax.imshow(errorImg, 'gray')
# plt.colorbar(cols)

# plt.show()



