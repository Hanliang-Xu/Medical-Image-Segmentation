import nrrd
from Project.volumeViewer import *
from Project.surface import *
import numpy as np
import matplotlib as mp
import nibabel as nib

# load a CT image to play with
img, imgh = nrrd.read('/Users/leonslaptop/Desktop/2024 Spring/ECE 3892/data/0522c0001/img.nrrd')
# Specify the path to your NIfTI file
# file_path = '/Users/leonslaptop/Desktop/2024 Spring/Imp0001-Decompressed_CT_0_2.nii'
# Load the NIfTI file
# nifti_file = nib.load(file_path)
# Get the data from the file
# img = nifti_file.get_fdata()

file_path = '/Users/leonslaptop/Desktop/2024 Spring/Research/Pelvis/head-NIFTI/head-Decompressed_CT_0_1.nii'
voxsz = [imgh['space directions'][0][0], imgh['space directions'][1][1], imgh['space directions'][2][2]]

d = volumeViewer()
d.setImage(img, voxsz, contrast=1500, level=500, autocontrast=False)
d.update(direction=2, slc=78)
# d.display()

imgzp = np.zeros(np.array(img.shape)+2)
imgzp[1:-1,1:-1,1:-1] = img

s = surface()
createSurfaceFromVolume(s, imgzp, voxsz, isolevel=700)

# undo zero padding
s.verts[:,0] -= voxsz[0]
s.verts[:,1] -= voxsz[1]
s.verts[:,2] -= voxsz[2]

buildGraph(s)
surfaces = connectedComponents(s)
numsurf = np.size(surfaces)
print(f'Found {numsurf} surfaces')

vols = np.zeros(numsurf)
for i in range(numsurf):
    vols[i] = volume(surfaces[i])

maxvol = np.max(vols)
imax = np.argmax(vols)
print(f'Surface {imax} has max volume {maxvol} mm3')

win2 = myVtkWin()

#show largest component in magenta
win2.addSurf(surfaces[imax].verts, surfaces[imax].faces, color=[1,0,1], opacity=1.0)

cols = mp.colormaps['jet']
for i in range(numsurf):
    if i!=imax and vols[i] > 1000:
        win2.addSurf(surfaces[i].verts, surfaces[i].faces,
                     color=cols(i % 256)[0:3], opacity=0.5)

win2.start()