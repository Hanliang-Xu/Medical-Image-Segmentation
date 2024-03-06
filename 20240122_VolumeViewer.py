import nrrd
import matplotlib.pyplot as plt
import numpy as np

# load a CT image to play with
img, imgh = nrrd.read('/Users/leonslaptop/Desktop/2024 Spring/ECE 3892/data/0522c0001/img.nrrd')
voxsz = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]

# print resolution and voxel size
print(f'Size of image: {np.shape(img)}')
print(f'Voxel size of image: {voxsz}')

#show a single axial slice
slc = 85
fig, ax = plt.subplots()
ax.imshow(img[:, :, slc])

#correct orientation and grayscale
ax.imshow(img[:,:,slc].T, 'gray')

#adjust intensity window
ax.imshow(img[:,:,slc].T, 'gray', vmin = -50, vmax=150)

#paramaterize window using 'level' and 'contrast' parameters
level = 0
contrast = 1000
ax.imshow(img[:,:,slc].T, 'gray', vmin = level-contrast/2, vmax=level+contrast/2)

# change from nearest neighbor to bilinear interpolation
ax.imshow(img[:,:,slc].T, 'gray', vmin = level-contrast/2, vmax=level+contrast/2,
          interpolation='bilinear')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Axial slice z = {slc}')

# Try coronal direction
slc = 256
fig, ax = plt.subplots()
ax.imshow(img[:,slc,:].T, 'gray', vmin = level-contrast/2, vmax=level+contrast/2,
          interpolation='bilinear')
ax.set_xlim(left=0, right=np.shape(img)[0]-1)
# make z-coordinates run from bottom to top
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
ax.set_aspect(voxsz[2] / voxsz[0])
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Coronal slice y = {slc}')


#Try sagittal direciton
slc = 256
fig, ax = plt.subplots()
ax.imshow(img[slc,:,:].T, 'gray', vmin = level-contrast/2, vmax=level+contrast/2,
          interpolation='bilinear')
ax.set_xlim(left=0, right=np.shape(img)[1]-1)
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
ax.set_aspect(voxsz[2] / voxsz[1])
plt.xlabel('y')
plt.ylabel('z')
plt.title(f'Sagittal slice x = {slc}')

# Add some user interface callback code
def onKeyPress(self, event):
    # event.inaxes == ax
    key = event.key
    direction = None  # Determine the active view based on some logic

    # Slice navigation
    if key in ['up', 'a'] and self.slc[direction] < np.shape(self.img)[direction] - 1:
        self.slc[direction] += 1
    elif key in ['down', 'z'] and self.slc[direction] > 0:
        self.slc[direction] -= 1

    # Window level adjustments
    if key == 'd':
        self.level += 0.1 * self.contrast
    elif key == 'x':
        self.level -= 0.1 * self.contrast

    # Contrast adjustments
    if key == 'c':
        self.contrast *= 1.1
    elif key == 'v':
        self.contrast *= 0.9

    self.updateDisplay()

plt.connect('key_press_event', onKeyPress)
plt.ion()

# integrate display into a class so callbacks have access to member variables
class displaySagittal:
    def __init__(self, img, voxsz, slc):
        self.img = img
        self.voxsz = voxsz
        self.slc = slc
        self.fig, self.ax = plt.subplots()
        plt.connect('key_press_event', self.onKeyPress)
        plt.ion()
        self.update(slc)

    def update(self, slc):
        self.slc = slc
        self.ax.imshow(self.img[slc,:,:].T,'gray',vmin=-500,vmax=500,
                  interpolation='bilinear')
        self.ax.set_xlim(left=0,right=np.shape(self.img)[1] - 1)
        self.ax.set_ylim(bottom=0,top=np.shape(self.img)[2] - 1)
        self.ax.set_aspect(self.voxsz[2] / self.voxsz[1])
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(f'Sagittal slice x = {self.slc}')

    def onKeyPress(self, event):
        if event.inaxes == self.ax:
            if event.key == 'a' and self.slc < np.shape(self.img)[0]-1:
                self.slc += 1
            elif event.key == 'z' and self.slc > 0:
                self.slc -= 1
        self.update(self.slc)

d = displaySagittal(img, voxsz, 256)
plt.show()


# Demo a volume viewer class that displays all three orthogonal views simultaneously
from Project.volumeViewer import *

v = volumeViewer()
v.setImage(img, voxsz)

# Other ways to analyze images beyond visual inspection -- use histograms to inspect intensity distributions
fig, ax = plt.subplots()
# plt.ioff()
# typically having 1 bin for each intensity is a bad idea because it makes the histogram too sparse
# plt.hist(img.flatten(), bins = np.linspace(np.amin(img), np.amax(img), (np.amax(img)-np.amin(img)+1).astype(int)))
plt.hist(img.flatten(), bins = np.linspace(np.amin(img), np.amax(img), 65))
plt.yscale('log')
plt.ylabel('# of occurrences')
plt.xlabel('Pixel intensities')
plt.show()


# estimate pdfs of two different regions
neck = img[:,:, 0:40]
head = img[:,:,40:]

bins = np.linspace(np.amin(img), np.amax(img), 65)
hist_neck, binsout = np.histogram(neck, bins=bins)
hist_head, _ = np.histogram(head, bins=bins)

pdf_neck = hist_neck/np.sum(hist_neck)
pdf_head = hist_head/np.sum(hist_head)

bin_centers = (bins[0:64] + bins[1:])/2
fig, ax = plt.subplots()
plt.plot(bin_centers, pdf_neck, color=[1,0,0])
plt.plot(bin_centers, pdf_head, color=[0,0,1])
plt.yscale('log')
plt.ylabel('Estimated probability')
plt.xlabel('Voxel intensity')
plt.show()

# Use multi-Otsu for automatic threshold selection
from skimage.filters import threshold_multiotsu

threshold = threshold_multiotsu(img, classes=2, nbins=64)
print(threshold)

thresholds = threshold_multiotsu(img, classes=3, nbins=64)
print(thresholds)


thresholds2 = threshold_multiotsu(img, classes=4, nbins=64)
print(thresholds2)

# v.display()
# display an isocontour at threshold levels
plt.figure(v.fig)
plt.axes(v.ax[1,0])

X,Y = np.meshgrid(range(np.shape(img)[0]), range(np.shape(img)[1]))
plt.contour(X, Y, img[:,:,v.slc[2]].T , levels=thresholds, colors = [[1,0,0], [0,1,0]])
plt.show()

while (1):
    d.fig.canvas.draw_idle()
    d.fig.canvas.start_event_loop(0.3)