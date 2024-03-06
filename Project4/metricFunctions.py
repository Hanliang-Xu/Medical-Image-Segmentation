import numpy as np
from tqdm import tqdm


def dice_coefficient(gt, pred):
  """
  Calculate the Dice coefficient, a measure of set similarity.

  Parameters:
  - gt: Ground truth binary segmentation mask as a numpy array.
  - pred: Predicted binary segmentation mask as a numpy array.

  Returns:
  - dice: Dice coefficient as a float.
  """
  # Calculate intersection and union
  intersection = np.logical_and(gt, pred).sum()
  gt_sum = gt.sum() + pred.sum()

  # Calculate Dice coefficient. Add a small epsilon to avoid division by zero.
  dice = (2. * intersection + 1e-6) / (gt_sum + 1e-6)

  return dice


def vectorized_distance_computation(points, triangles):
  """
    Computes the minimum distance from each point to any of the given triangles, integrating decision making
    for on-surface or off-surface projection.

    Args:
        points (np.array): Array of shape (N, 3) containing N points in 3D space.
        triangles (np.array): Array of shape (M, 3, 3) representing M triangles, each defined by 3 vertices.
        norms (np.array): Array of shape (M, 3) containing the normal vectors for M triangles.

    Returns:
        np.array: Array of shape (N,) containing the minimum distance from each point to the closest triangle.
    """
  N = len(points)
  M = len(triangles)

  distances = np.inf * np.ones((N, M))

  # Calculate normals for each triangle
  edge1 = triangles[:, 1, :] - triangles[:, 0, :]
  edge2 = triangles[:, 2, :] - triangles[:, 0, :]
  norms = np.cross(edge1, edge2)
  norms_magnitude = np.linalg.norm(norms, axis=1, keepdims=True)
  norms = norms / norms_magnitude  # Normalize the normals

  # Iterate over each triangle with progress updates
  for j in tqdm(range(M), desc="Computing distances to triangles"):
    triangle = triangles[j]
    norm = norms[j]

    # Compute projected points and check if inside or outside the triangle
    projected_points, is_inside = project_and_check(points, triangle, norm)

    # Initialize an array to store the minimum distances for this triangle
    min_distances_for_triangle = np.zeros(N)

    if np.any(is_inside):
      points_inside = points[is_inside]
      projected_points_inside = projected_points[is_inside]
      distances_to_plane = calculate_distances_from_projected_points(points_inside,
                                                                     projected_points_inside)
      min_distances_for_triangle[is_inside] = distances_to_plane

    # Compute distances for points projecting off the surface (edges and vertices)
    if np.any(~is_inside):
      points_outside = points[~is_inside]
      edge_distances = vectorized_distance_to_edges(points_outside, triangle)
      min_distances_for_triangle[~is_inside] = np.min(edge_distances, axis=1)

    # Update the distances matrix for this triangle
    distances[:, j] = min_distances_for_triangle

  return distances


def calculate_distances_from_projected_points(points, projected_points):
  """
  Computes the distances from points to their projections on the plane.

  Args:
      points (np.array): Array of shape (N, 3) containing N points in 3D space.
      projected_points (np.array): Array of shape (N, 3) containing the projections of points onto a plane.

  Returns:
      np.array: Distances from each point to its projection on the plane.
  """
  # Calculate the vector differences between points and their projections
  vector_differences = points - projected_points

  # Compute the distances as the norm of these vector differences
  distances = np.linalg.norm(vector_differences, axis=1)

  return distances


def project_and_check(points, triangle, norm):
  # Calculate the vector from the triangle's first vertex to the points
  v0 = triangle[0]
  vectors_to_points = points - v0

  # Calculate the distance from the points to the triangle plane
  distance_to_plane = np.dot(vectors_to_points, norm)

  # Project points onto the plane
  projected_points = points - np.outer(distance_to_plane, norm)

  # Check if the projected points are inside the triangle
  is_inside = is_point_inside_triangle(projected_points, triangle)

  return projected_points, is_inside


def is_point_inside_triangle(pts, tri):
  # Barycentric technique to check if point is inside the triangle
  v0, v1, v2 = tri[0], tri[1], tri[2]
  v0v1 = v1 - v0
  v0v2 = v2 - v0

  # Prepare points
  v0p = pts - v0[np.newaxis, :]  # Vector from v0 to each point

  # Compute dot products
  dot00 = np.dot(v0v2, v0v2)
  dot01 = np.dot(v0v1, v0v2)
  dot11 = np.dot(v0v1, v0v1)
  dot02 = np.einsum('ij,j->i', v0p, v0v2)  # Dot product of v0p vectors with v0v2
  dot12 = np.einsum('ij,j->i', v0p, v0v1)  # Dot product of v0p vectors with v0v1

  # Compute barycentric coordinates
  invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
  u = (dot11 * dot02 - dot01 * dot12) * invDenom
  v = (dot00 * dot12 - dot01 * dot02) * invDenom

  # Check if point is in triangle
  inside = (u >= 0) & (v >= 0) & (u + v <= 1)

  return inside


def vectorized_distance_to_edges(points, triangle):
  """
  Computes the vectorized distance from points to the edges of a triangle.

  Args:
      points (np.array): Array of shape (N, 3) containing N points.
      triangle (np.array): Array of shape (3, 3) representing a triangle's vertices.

  Returns:
      np.array: Array of shape (N, 3) containing distances from each point to each of the triangle's edges.
  """
  # Calculate edge vectors
  edges = np.array(
    [triangle[1] - triangle[0], triangle[2] - triangle[1], triangle[0] - triangle[2]])

  # Vector from each vertex to points
  p_to_vertices = np.array([points - triangle[0], points - triangle[1], points - triangle[2]])

  # Calculate projection coefficients for all edges
  coefficients = np.einsum('ijk,ik->ij', p_to_vertices, edges) / (
      np.linalg.norm(edges, axis=1)[:, np.newaxis] ** 2)

  # Clip coefficients to [0, 1] range
  coefficients_clipped = np.clip(coefficients, 0, 1)

  # Ensure coefficients are correctly broadcastable to (3, 8671, 3) for multiplication with edges
  coefficients_reshaped = coefficients_clipped[:, :, np.newaxis]

  # Broadcast triangle for addition: reshape triangle for compatibility
  triangle_broadcast = triangle[:, np.newaxis, :]  # Reshape to (3, 1, 3) for broadcasting

  # Now apply the reshaped coefficients to edges and add the broadcasted triangle vertices
  projections = triangle_broadcast + coefficients_reshaped * edges[:, np.newaxis, :]

  # Calculate distances from points to projected points
  distances = np.sqrt(np.sum((points[np.newaxis, :, :] - projections) ** 2, axis=2))

  # Stack distances into a single array
  distances = distances.T

  return distances
