import nrrd

from Project.surface import *
from Project4.confusionMatrix import *
from Project4.metricFunctions import *
from Project4.plotStatistics import *

# Assuming the base directory and a dictionary of patients are defined
bsdir = '/Users/leonslaptop/Desktop/2024 Spring/ECE 3892/data/'

# Lists to store metrics for each rater and the majority vote
metrics = {
  'volume': {'t1': [], 't2': [], 't3': [], 'mv': []},
  'dice': {'t1': [], 't2': [], 't3': [], 'mv': []},
  'mean_surface_distance': {'t1': [], 't2': [], 't3': [], 'mv': []},
  'hausdorff_distance': {'t1': [], 't2': [], 't3': [], 'mv': []}
}

# Initialize dictionaries to hold the cumulative confusion matrices for each rater
cumulative_matrices = {
  't1': confusionMatrix(np.array([]), np.array([])),
  't2': confusionMatrix(np.array([]), np.array([])),
  't3': confusionMatrix(np.array([]), np.array([]))
}

pts = {0: '0522c0001', 1: '0522c0002', 2: '0522c0003', 3: '0522c0009', 4: '0522c0013',
       5: '0522c0014', 6: '0522c0017', 7: '0522c0057', 8: '0522c0070', 9: '0522c0077'}

# Loop through the first 10 cases
for pt_id, pt in pts.items():
  gt_path = f'{bsdir}{pt}/structures/mandible.nrrd'
  gt, hdr = nrrd.read(gt_path)
  voxsz = [hdr['space directions'][0][0], hdr['space directions'][1][1],
           hdr['space directions'][2][2]]

  # Load segmentations
  t1, _ = nrrd.read(f'{bsdir}{pt}/structures/target1.nrrd')
  t2, _ = nrrd.read(f'{bsdir}{pt}/structures/target2.nrrd')
  t3, _ = nrrd.read(f'{bsdir}{pt}/structures/target3.nrrd')

  # Majority vote
  mv = np.sum([t1, t2, t3], axis=0) > 1.5

  # Create surfaces for each segmentation
  surfaces = {
    'gt': surface(),
    't1': surface(),
    't2': surface(),
    't3': surface(),
    'mv': surface()
  }
  surfaces['gt'].createSurfaceFromVolume(gt, voxsz, 0.5)
  surfaces['t1'].createSurfaceFromVolume(t1, voxsz, 0.5)
  surfaces['t2'].createSurfaceFromVolume(t2, voxsz, 0.5)
  surfaces['t3'].createSurfaceFromVolume(t3, voxsz, 0.5)
  surfaces['mv'].createSurfaceFromVolume(mv.astype(int), voxsz, 0.5)

  # VTK Visualization (Optional: Uncomment to use if myVtkWin is configured correctly)
  win = myVtkWin()
  win.addSurf(surfaces['t1'].verts, surfaces['t1'].faces, opacity=0.5)
  win.addSurf(surfaces['t2'].verts, surfaces['t2'].faces, color=[1, 1, 0], opacity=0.5)
  win.addSurf(surfaces['t3'].verts, surfaces['t3'].faces, color=[1, 0, 1], opacity=0.5)
  # win.start()

  # Calculate metrics for each rater and the majority vote
  segmentations = {'t1': t1, 't2': t2, 't3': t3, 'mv': mv.astype(int)}

  for rater, seg in segmentations.items():
    dice_score = dice_coefficient(gt, seg)
    metrics['dice'][rater].append(dice_score)

    volume = surfaces[rater].volume()
    metrics['volume'][rater].append(volume)

    MASD_gt_rater, HD_gt_rater, _, _ = surfaces['gt'].surfDistances(surfaces[rater])
    MASD_rater_gt, HD_rater_gt, _, _ = surfaces[rater].surfDistances(surfaces['gt'])
    mean_surface_distance = (MASD_gt_rater + MASD_rater_gt) / 2
    hausdorff_distance = max(HD_gt_rater, HD_rater_gt)

    metrics['mean_surface_distance'][rater].append(mean_surface_distance)
    metrics['hausdorff_distance'][rater].append(hausdorff_distance)

  print(f"{pt} DONE!")

  # Temporarily create confusion matrices for current patient and rater to update cumulative matrices
  for rater, segmentation in [('t1', t1), ('t2', t2), ('t3', t3)]:
    current_matrix = confusionMatrix(gt, segmentation)
    cumulative_matrices[rater].TP += current_matrix.TP
    cumulative_matrices[rater].FP += current_matrix.FP
    cumulative_matrices[rater].FN += current_matrix.FN
    cumulative_matrices[rater].TN += current_matrix.TN

# Calculate and print sensitivity and specificity for each rater
sensitivity_specificity = {}

# After looping through all patients, calculate and print out the overall metrics for each rater
for rater, matrix in cumulative_matrices.items():
  print(f"Confusion matrix for {rater}:")
  matrix.print()  # This now uses the print method of the confusionMatrix class

  sensitivity = matrix.sensitivity()
  specificity = matrix.specificity()

  print(f"{rater}: Sensitivity = {sensitivity:.4f}, Specificity = {specificity:.4f}")
  print(f"Dice: {matrix.dice()}\n")

plot_metrics(metrics, "Segmentation Metrics Across 10 Cases")

# Perform and print Wilcoxon signed-rank test results
perform_wilcoxon_test(metrics, 'volume')
perform_wilcoxon_test(metrics, 'dice')
perform_wilcoxon_test(metrics, 'mean_surface_distance')
perform_wilcoxon_test(metrics, 'hausdorff_distance')

overall_metrics = calculate_overall_metrics(cumulative_matrices)

# Print the overall confusion matrices and sensitivity/specificity values
for rater, metrics in overall_metrics.items():
  print(f"Rater: {rater}")
  print("Confusion Matrix:")
  cumulative_matrices[rater].print()
  print(f"Sensitivity: {metrics['Sensitivity']:.3f}")
  print(f"Specificity: {metrics['Specificity']:.3f}\n")
