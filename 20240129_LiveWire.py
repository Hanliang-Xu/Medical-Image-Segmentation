import numpy as np

from LiveWire import *
from LiveWire3D import *
import nrrd
from Project.volumeViewer import *
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import feature

# load a CT image to play with
img, imgh = nrrd.read('C:\\Users\\noblejh\\Box Sync\\EECE_395\\0522c0001\\img.nrrd')
voxsz = np.array([imgh['space directions'][0][0], imgh['space directions'][1][1],
         imgh['space directions'][2][2]])
voxszo = np.copy(voxsz)

# crop and downsample
fp = np.array([200, 130, 52])-1 # right, anterior, inferior corner
sp = np.array([320, 240, 77]) # left, posterior, superior corner
crp = img[fp[0]:sp[0]:2, fp[1]:sp[1]:2, fp[2]:sp[2]:2]
voxsz *= 2

# v = volumeViewerp1()
# v.setImage(crp, voxsz)
# v.display()

# setup edge graph for single slice
slc = 5
img2d = crp[:,:,slc]

r,c = np.shape(img2d)
sobel = np.zeros((r, c, 4))
sobel[:,:,0] = ndi.convolve(img2d, np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])) # x-y-/x+y+
sobel[:,:,1] = ndi.convolve(img2d, np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])) # x-y+/x+y-
sobel[:,:,2] = ndi.convolve(img2d, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])) # y+/-
sobel[:,:,3] = ndi.convolve(img2d, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T) # x+/-

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(sobel[:,:,0].T, 'gray')
ax[0,1].imshow(sobel[:,:,1].T, 'gray')
ax[1,0].imshow(sobel[:,:,2].T, 'gray')
ax[1,1].imshow(sobel[:,:,3].T, 'gray')
# plt.show()

# invert Sobel for low cost on edges
sobel = np.abs(sobel)
for i in range(4):
    sobel[:,:,i] = 1 - sobel[:,:,i]/np.amax(sobel[:,:,i])

ax[0,0].imshow(sobel[:,:,0].T, 'gray')
ax[0,1].imshow(sobel[:,:,1].T, 'gray')
ax[1,0].imshow(sobel[:,:,2].T, 'gray')
ax[1,1].imshow(sobel[:,:,3].T, 'gray')
# plt.show()

canny = 1 - feature.canny(img2d, sigma=1)
fig, ax = plt.subplots()
ax.imshow(canny.T, 'gray')
# plt.show()

# mixture of two
fig, ax = plt.subplots()
ax.imshow((canny*0.5 + sobel[:,:,2]).T, 'gray')
# plt.show()


edges = [[] for i in range(np.size(img2d))]
alpha = 0.5 # canny vs sobel
beta = 0.1 # smoothness/straightness

sq2 = np.sqrt(2)
for x in range(r):
    for y in range(c):
        pr = x + y*r
        if x > 0:
            if y > 0:
                nd = (x - 1 + (y-1) * r)# diagonal x-y- neib
                edges[pr].append(lwnode(nd=nd, pr=pr, cost=
                    sq2 * (beta +  alpha* canny[x, y] + (1-alpha)*sobel[x,y,0])))

            if y < c-1:
                nd = (x - 1 + (y+1) * r)# diagonal x-y+ neib
                edges[pr].append(lwnode(nd=nd, pr=pr, cost=
                    sq2 * (beta +  alpha* canny[x, y] + (1-alpha)*sobel[x,y,1])))

            nd = (x - 1 + (y) * r)  #  x- neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3])))

        if x < r-1:
            if y>0:
                nd = (x + 1 + (y - 1) * r)  # diagonal x+y- neib
                edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                sq2 * (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1])))

            if y<c - 1:
                nd = (x + 1 + (y + 1) * r)  # diagonal x+y+ neib
                edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                sq2 * (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,0])))

            nd = (x + 1 + (y) * r)  #  x+ neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3])))

        if y> 0:
            nd = (x + (y-1) * r)  #  y- neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2])))
        if y < c-1:
            nd = (x + (y+1) * r)  #  y+ neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2])))

fig, ax = plt.subplots()
ax.imshow(img2d.T, 'gray')
plt.ion()
plt.show()
lw = liveWire(img2d, edges=edges)

print(lw.cntrcur)


# let's see how heap works
import heapq as hq
h = []
# first we need a class for the node datatype that the heap will sort

class lwnode:
    def __init__(self, ncopy=None, par=None, nd=-1, pr=-1, cost=0):
        self.nd = nd
        self.pr = pr
        self.cost = cost

        if ncopy is not None: # ncopy is lwnode
            self.nd = ncopy.nd
            self.pr = ncopy.pr
            self.cost = ncopy.cost

        if par is not None: # par is lwnode
            self.par = par.nd
            self.cost += par.cost

    # define this for heap sorting:
    def __lt__(self, rhs):
        return self.cost < rhs.cost

nd = lwnode(nd=0, pr=-1, cost=1)

hq.heappush( h, nd)

hq.heappush(h, lwnode(nd=1, pr=0, cost=20))
hq.heappush(h, lwnode(nd=2, pr=0, cost=2))
hq.heappush(h, lwnode(nd=3, pr=0, cost=3))
hq.heappush(h, lwnode(nd=4, pr=2, cost=9))
hq.heappush(h, lwnode(nd=5, pr=1, cost=23))
hq.heappush(h, lwnode(nd=6, pr=1, cost=21))
hq.heappush(h, lwnode(nd=7, pr=2, cost=5))
hq.heappush(h, lwnode(nd=8, pr=6, cost=22))

for i in range(len(h)):
    print(h[i].cost)
# can see min at front

n = hq.heappop(h)
print(f'Popped node with nd={n.nd} and cost={n.cost}')
for i in range(len(h)):
    print(h[i].cost)

print(' ')
hq.heappush(h, lwnode(nd=9, pr=3, cost=4))
for i in range(len(h)):
    print(h[i].cost)
# same heap as in lecture slides...

# next, graphSearch functions
# class graphSearch:
#   def __init__(self, node_type):
#   def run(self, seed, endnode=None):
        #...
        # return self.trace(lastnode.nd, seed), lastnode.cost
#   def trace(self, nd, seed)

import copy
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

# try on small example
r = 50
c = 60
alpha = 0.25
beta = 0.1
X,Y = np.meshgrid(range(c), range(r))
img2d = np.zeros((r,c)) + 10*((X - c/2)*(X-c/2) + (Y-r/2)*(Y-r/2) < 400)
img2d += np.random.default_rng(0).normal(size=(r,c))*8

seed = 2 + 2*r
endnode = 47 + 57*r

fig, ax = plt.subplots()
ax.imshow(img2d.T, 'gray')
plt.ioff()
# plt.show()


r,c = np.shape(img2d)
sobel = np.zeros((r, c, 4))
sobel[:,:,0] = ndi.convolve(img2d, np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])) # x-y-/x+y+
sobel[:,:,1] = ndi.convolve(img2d, np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])) # x-y+/x+y-
sobel[:,:,2] = ndi.convolve(img2d, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])) # y+/-
sobel[:,:,3] = ndi.convolve(img2d, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T) # x+/-

# invert Sobel for low cost on edges
sobel = np.abs(sobel)
for i in range(4):
    sobel[:,:,i] = 1 - sobel[:,:,i]/np.amax(sobel[:,:,i])

canny = 1 - feature.canny(img2d, sigma=1)

edges = [[] for i in range(np.size(img2d))]

sq2 = np.sqrt(2)
for x in range(r):
    for y in range(c):
        pr = x + y*r
        if x > 0:
            if y > 0:
                nd = (x - 1 + (y-1) * r)# diagonal x-y- neib
                edges[pr].append(lwnode(nd=nd, pr=pr, cost=
                    sq2 * (beta +  alpha* canny[x, y] + (1-alpha)*sobel[x,y,0])))

            if y < c-1:
                nd = (x - 1 + (y+1) * r)# diagonal x-y+ neib
                edges[pr].append(lwnode(nd=nd, pr=pr, cost=
                    sq2 * (beta +  alpha* canny[x, y] + (1-alpha)*sobel[x,y,1])))

            nd = (x - 1 + (y) * r)  #  x- neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3])))

        if x < r-1:
            if y>0:
                nd = (x + 1 + (y - 1) * r)  # diagonal x+y- neib
                edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                sq2 * (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,1])))

            if y<c - 1:
                nd = (x + 1 + (y + 1) * r)  # diagonal x+y+ neib
                edges[pr].append(lwnode(nd=nd,pr=pr,cost=
                sq2 * (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,0])))

            nd = (x + 1 + (y) * r)  #  x+ neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,3])))

        if y> 0:
            nd = (x + (y-1) * r)  #  y- neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2])))
        if y < c-1:
            nd = (x + (y+1) * r)  #  y+ neib
            edges[pr].append(lwnode(nd=nd,pr=pr,cost=
            (beta + alpha * canny[x,y] + (1 - alpha) * sobel[x,y,2])))

# run the graph search
# gs = graphSearchLW()
# pathnodes, pathcost = gs.run(edges, seed, endnode)
# print(pathcost)
#
# #extract x-y coordinates from node numbers
# path = np.zeros((np.size(pathnodes), 2))
# for i in range(np.size(pathnodes)):
#     path[i,0] = pathnodes[i] % r
#     path[i,1] = pathnodes[i]//r
#
# # display the result
# plt.plot(path[:,0], path[:,1], 'm')
# plt.show()

plt.close('all')
crp = img[fp[0]:sp[0], fp[1]:sp[1], fp[2]:sp[2]]
lw3d = liveWire3D(crp, voxsz)

cntr = lw3d.cntr

win  = myVtkWin()
for i in range(len(cntr)):
    for j in range(len(cntr[i])):
        win.addLines(np.concatenate(
            (cntr[i][j][:, 0:2]*voxszo[0:2][np.newaxis,:], np.repeat(i*voxszo[2],
                len(cntr[i][j]))[:,np.newaxis]), axis=1),
            np.concatenate((np.arange(len(cntr[i][j]))[:,np.newaxis],
                            ((np.arange(len(cntr[i][j]))+1)%len(cntr[i][j]))[:,np.newaxis]), axis=1),
            color=[0,1,0])

win.start()


# defining interaction class
class liveWire():
    def __init__(self, img, alpha=0.5, beta = 0.1, edges=None):
        pass# remove when this function is coded
        # set member variables using inputs and initialize other member variables
        # self.alpha, beta, lastseed, gs, img, edges, r, c, cntrcur, ax, exit
        # gs is graphSearchLW object
        # cntrcur contains finished contour segments
        # ax is handle to current axes
        # exit is a flag to exit when we are finished

        # connect callbacks (key presses, mouse clicks, mouse movements)
        # auto-compute edge weights using sobel/canny with input alpha, beta if no edges given
        # run while loop while user is live-wiring

    def display(self):
        pass # remove when this function is coded
        # update image display with already finalized contour segments in self.cntrcur in green,
        #     if any exist

    def on_mouse_move(self, event):
        pass # remove when this function is coded
        # if the mouse is in the display axes area and a contour has been started,
        # repaint the image and display the candidate path that leads from the last seed to
        # the current mouse position in red

    def on_mouse_click(self, event):
        if event.button is MouseButton.RIGHT:
            if np.size(self.cntrcur) > 0:
                x = np.round(event.xdata).astype(np.longlong)
                y = np.round(event.ydata).astype(np.longlong)
                nextseed = x + self.r*y
                self.addPath(nextseed, self.lastseed)
                self.exit=1

            self.display()

        elif event.button is MouseButton.LEFT:
            x = np.round(event.xdata).astype(np.longlong)
            y = np.round(event.ydata).astype(np.longlong)
            if self.lastseed is None:
                self.lastseed = x + self.r*y
                self.gs.run(self.edges, self.lastseed)
            else:
                # finished one contour and need to trace it back
                nextseed = x + self.r*y
                self.addPath(nextseed, self.lastseed)
                self.lastseed = nextseed
                self.gs.run(self.edges, self.lastseed)

    def addPath(self, endpoint, seed):
        # add current path to final contour
        # don't need seed because it was endpoint of last path in contour
        newpath = np.flip(self.gs.trace(endpoint, seed))[1:]
        newpathxy = np.zeros((np.size(newpath), 2))
        newpathxy[:,0] = newpath % self.r
        newpathxy[:,1] = newpath // self.r
        if np.size(self.cntrcur) == 0:
            self.cntrcur = np.concatenate(([[seed%self.r, seed//self.r]], newpathxy), axis=0)
        else:
            self.cntrcur = np.concatenate((self.cntrcur, newpathxy), axis=0)
