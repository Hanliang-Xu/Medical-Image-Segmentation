#Introduction to visualization with VTK
import vtk
import numpy as np

# start with simple visualization of 2 points
verts = np.array([[0.,0.,0], [1., 1., 1.]])

pnts = vtk.vtkPoints()
for j, p in enumerate(verts):
    pnts.InsertPoint(j, p)

# cells used to define arbitrary object vertex connections
cells = vtk.vtkCellArray()
for j in range(pnts.GetNumberOfPoints()):
    vil = vtk.vtkIdList()
    vil.InsertNextId(j)
    cells.InsertNextCell(vil)

poly = vtk.vtkPolyData()
poly.SetPoints(pnts)
poly.SetVerts(cells)

# mapper takes arbitrary polydata as input
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly)

# each object (polydata, imagedata, etc in VTK has its own actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0,1,1)
actor.GetProperty().SetOpacity(1)
actor.GetProperty().SetPointSize(4)

renwin = vtk.vtkRenderWindow() # creates a new popup window
ren = vtk.vtkRenderer() # Create a 3D renderer for the window
renwin.AddRenderer(ren)
inter = vtk.vtkRenderWindowInteractor() # makes renderer interactive
inter.SetRenderWindow(renwin)
inter.Initialize()
inter.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera()) # Set interaction style

ren.AddActor(actor)
# start interactive view (blocking, press q to quit)
inter.Start()