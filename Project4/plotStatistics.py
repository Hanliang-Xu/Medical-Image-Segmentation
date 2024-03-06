import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


# Assuming dice_scores_all is a dictionary with rater keys ('t1', 't2', 't3', 'mv') and lists of Dice scores as values
# Example:
# dice_scores_all = {
#     't1': [0.9, 0.92, 0.93, ...],  # Dice scores for rater 1 across all cases
#     't2': [0.91, 0.9, 0.92, ...],  # Dice scores for rater 2 across all cases
#     't3': [0.88, 0.89, 0.9, ...],  # Dice scores for rater 3 across all cases
#     'mv': [0.94, 0.95, 0.96, ...],  # Dice scores for majority vote across all cases
# }

# Visualization with boxplots for each metric
def plot_metrics(metric_data, title):
  fig, axs = plt.subplots(1, len(metric_data), figsize=(20, 5))
  for ax, (metric_name, values) in zip(axs, metric_data.items()):
    ax.boxplot(values.values(), labels=values.keys())
    ax.set_title(f'{title}: {metric_name}')
    ax.set_ylabel(metric_name)
    ax.grid(True)
  plt.tight_layout()
  plt.show()


# Function to compare metrics using Wilcoxon signed-rank test
def compare_metrics(metric_data):
  raters = list(metric_data.keys())
  p_values_table = np.zeros((len(raters), len(raters)), dtype=float)

  # Fill the table with p-values
  for i, rater1 in enumerate(raters):
    for j, rater2 in enumerate(raters):
      if i < j:  # Avoid redundant comparisons
        stat, p_value = wilcoxon(metric_data[rater1], metric_data[rater2])
        p_values_table[i, j] = p_value
        p_values_table[j, i] = p_value  # Symmetric matrix
      elif i == j:
        p_values_table[i, j] = np.nan  # NaN for comparisons with themselves

  return p_values_table, raters


# Function to perform Wilcoxon signed-rank test and print results
def perform_wilcoxon_test(metrics, metric_name):
  raters = list(metrics[metric_name].keys())
  num_raters = len(raters)
  p_values_table = np.empty((num_raters, num_raters))
  p_values_table[:] = np.NaN  # Initialize with NaN

  # Perform Wilcoxon signed-rank test between pairs of raters for the specified metric
  for i in range(num_raters):
    for j in range(i + 1, num_raters):
      scores_i = metrics[metric_name][raters[i]]
      scores_j = metrics[metric_name][raters[j]]
      stat, p_value = wilcoxon(scores_i, scores_j)
      p_values_table[i, j] = p_value
      p_values_table[j, i] = p_value  # Symmetric matrix

  # Print the table of p-values
  print(f"Wilcoxon signed-rank test p-values for {metric_name}:")
  print("\t" + "\t".join(raters))
  for i, rater in enumerate(raters):
    print(f"{rater}\t" + "\t".join(
      ["{:.3f}".format(p) if not np.isnan(p) else "NaN" for p in p_values_table[i]]))

  # Identify significant differences
  alpha = 0.05
  print(f"\nSignificant differences for {metric_name} (p < {alpha}):")
  for i in range(num_raters):
    for j in range(i + 1, num_raters):
      if p_values_table[i, j] < alpha:
        print(f"Between {raters[i]} and {raters[j]}: YES (p = {p_values_table[i, j]:.3f})")
      # else:
        # print(f"Between {raters[i]} and {raters[j]}: NO (p = {p_values_table[i, j]:.3f})")


# Assuming cumulative_matrices is filled as before, calculate sensitivity and specificity for each rater
def calculate_overall_metrics(cumulative_matrices):
  results = {}
  for rater, cm in cumulative_matrices.items():
    sensitivity = cm.sensitivity()
    specificity = cm.specificity()
    results[rater] = {'Sensitivity': sensitivity, 'Specificity': specificity}
  return results
