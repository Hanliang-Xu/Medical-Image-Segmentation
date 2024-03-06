from scipy.spatial import KDTree

from Project.volumeViewer import *
from Project4.metricFunctions import *


class GraphNode:
  def __init__(self, vertex_id):
    self.id = vertex_id
    self.neighbors = []


# surface class
class surface:
  def __init__(self):
    self.verts = []
    self.faces = []
    self.graph = []

  def createSurfaceFromVolume(self, img, voxsz, isolevel=700):
    from skimage import measure
    # Use marching cubes to generate vertices and faces and assign generated vertices and faces to class variables
    self.verts, self.faces, _, _ = measure.marching_cubes(img, level=isolevel, spacing=voxsz)

  def volume(self):
    # Ensure verts and faces are not None
    if self.verts is None or self.faces is None:
      raise ValueError("Surface vertices and faces must be defined.")

    # Calculate volume
    volume = 0.0
    for face in self.faces:
      v0, v1, v2 = self.verts[face]
      # The volume contribution of the tetrahedron formed by face and origin
      tetra_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
      volume += tetra_volume
    # Absolute value to ensure positive volume, in case of inverted normals
    return abs(volume)

  def surfDistances(self, mesh2):
    """
    Calculate surface distances from mesh1 to mesh2, integrating tqdm for progress tracking.
    """

    triangles = np.array(
      [mesh2.verts[face_indices] for face_indices in mesh2.faces])  # Shape (M, 3, 3)

    # Placeholder for the actual vectorized computation
    distances = vectorized_distance_computation(self.verts, triangles)

    # Step 3: Minimization
    min_distances = np.min(distances, axis=1)  # Assuming distances is of shape (N, M)
    mean_distance = np.mean(min_distances)
    max_distance = np.max(min_distances)

    return mean_distance, max_distance, None, None

  def pointsetDistance(self, t1s):
    """
    Calculate point set distances between two sets of points, including Mean Absolute Point Set Distance (MAPD),
    Hausdorff Point Distance (HPD), and points of interest related to HPD.

    :param gts: Ground Truth Set, with points as an Nx3 numpy array.
    :param t1s: Target Set 1, with points as an Mx3 numpy array.
    :return: MAPD from gts to t1s, HPD from gts to t1s, and points of interest.
    """
    # Build a KD-tree for the vertices of mesh2
    tree = KDTree(t1s.verts)

    # Query the KD-tree for the closest point in mesh2 for each vertex in mesh1
    distances, _ = tree.query(self.verts)

    # Calculate MASD and HD
    MASDg1 = np.mean(distances)

    HDg1 = np.max(distances)

    return None, MASDg1, HDg1


def demoSurfaceFromNRRD():
  import nrrd
  from skimage import measure

  # load CT image
  img, header = nrrd.read('/data/0522c0001/img.nrrd')

  # isosurface it at isolevel =700 to separate bone from soft-tissue/air
  # When isosurfacing a binary segmentation mask, often an isolevel=0.5 is used
  s = surface()
  s.verts, s.faces, _, _ = measure.marching_cubes(img, level=-300)

  # display result in myVtkWin
  win = myVtkWin()
  win.addSurf(s.verts, s.faces, color=[1, .9, .8])
  win.start()

  # create surface accounting for anisotropic voxel size
  voxsz = [header['space directions'][0][0], header['space directions'][1][1],
           header['space directions'][2][2]]  # mm/voxel
  s.verts, s.faces, _, _ = measure.marching_cubes(img, level=700, spacing=voxsz)

  win = myVtkWin()
  win.addSurf(s.verts, s.faces, color=[1, .9, .8])
  win.start()


def projectOneTaskOne():
  # Initialize visualization window
  win = myVtkWin(title="Project One Task One ")

  # Define file paths and isolevels
  structures = [
    ("data/0522c0001/structures/brainstem.nrrd", 0, [1.0, 0.0, 0.0]),  # Red
    ("data/0522c0001/structures/OpticNerve_L.nrrd", 0, [0.0, 1.0, 0.0]),  # Green
    ("data/0522c0001/structures/OpticNerve_R.nrrd", 0, [0.0, 0.0, 1.0]),  # Blue
    ("data/0522c0001/structures/chiasm.nrrd", 0, [1.0, 1.0, 0.0]),  # Yellow
    ("data/0522c0001/structures/mandible.nrrd", 0, [0.0, 1.0, 1.0])  # Cyan
  ]

  # Process and display each structure
  for filePath, isolevel, color in structures:
    s = loadAndProcessStructure(filePath, isolevel)
    win.addSurf(s.verts, s.faces, color=color, opacity=1.0)

  # Finalize and start the visualization
  win.cameraPosition(position=[0, -800, 0], viewup=[0, 0, 1])
  win.start()


def loadAndProcessStructure(filePath, isolevel):
  import nrrd
  # Load NRRD file
  img, header = nrrd.read(filePath)
  voxsz = [header['space directions'][0][0], header['space directions'][1][1],
           header['space directions'][2][2]]  # mm/voxel

  # Create surface
  s = surface()
  createSurfaceFromVolume(s, img, voxsz, isolevel)
  return s


# Function to visualize the surface using VTK
def visualizeSurface(s):
  win = myVtkWin()
  win.addSurf(s.verts, s.faces, color=[1, 0.9, 0.8])
  win.start()


def buildGraph(self):
  # Initialize nodes for all vertices
  for i in range(len(self.verts)):
    self.graph.append(GraphNode(i))

  # Add edges based on faces
  for face in self.faces:
    for i, vertex_id in enumerate(face):
      # Add edge between current vertex and the next vertex in the face (forming edges of the triangle)
      next_vertex_id = face[(i + 1) % len(face)]
      if next_vertex_id not in self.graph[vertex_id].neighbors:
        self.graph[vertex_id].neighbors.append(next_vertex_id)
        self.graph[next_vertex_id].neighbors.append(vertex_id)


def connectedComponents(self):
  # Initialize Marked, labels, maxlabel=0
  num_vertices = len(self.verts)
  Marked = [False] * num_vertices
  labels = [-1] * num_vertices
  maxlabel = 0

  nodes = self.graph

  # Function for graph traversal and marking the connected component
  def markComponent(start):
    nonlocal maxlabel
    queue = [start]
    labels[start] = maxlabel
    Marked[start] = True
    while queue:
      current_vertex_id = queue.pop(0)
      current_node = nodes[current_vertex_id]
      for neighbor in current_node.neighbors:
        if not Marked[neighbor]:
          Marked[neighbor] = True
          labels[neighbor] = maxlabel
          queue.append(neighbor)

  # While there are unmarked vertices, perform a graph traversal
  for n in range(num_vertices):
    if not Marked[n]:
      markComponent(n)
      maxlabel += 1

  # Initialize containers
  label_to_vertices = {i: [] for i in range(max(labels) + 1)}
  label_to_faces = {i: [] for i in range(max(labels) + 1)}

  # Map vertices to their labels
  for vertex_index, label in enumerate(labels):
    label_to_vertices[label].append(vertex_index)

  # Iterate over faces to map them to labels
  for face_index, face in enumerate(self.faces):
    vertex_label = labels[face[0]]  # Assuming all vertices in a face share the same label
    label_to_faces[vertex_label].append(face_index)

  # Create components
  H = []
  for label, verts_indices in label_to_vertices.items():
    faces_indices = label_to_faces[label]

    # Assuming createComponent can handle lists of indices directly
    component = createComponent(self, verts_indices, faces_indices)
    H.append(component)
  return H


def createComponent(surfaceObj, verts_indices, faces_indices):
  new_component = surface()

  # Directly use numpy array, avoid converting to list unless necessary for downstream operations
  new_component.verts = surfaceObj.verts[verts_indices]

  # Mapping remains efficient as is
  new_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(verts_indices)}

  # Remap faces to new vertex indices, this part is already quite efficient
  remapped_faces = [[new_indices_map[vertex] for vertex in face] for face in
                    surfaceObj.faces[faces_indices]]
  new_component.faces = remapped_faces
  return new_component
