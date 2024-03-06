import numpy as np


class confusionMatrix:
  def __init__(self, gt, pred):
    """Initialize confusion matrix with ground truth and prediction."""
    self.TP = np.sum((gt == 1) & (pred == 1))
    self.FP = np.sum((gt == 0) & (pred == 1))
    self.TN = np.sum((gt == 0) & (pred == 0))
    self.FN = np.sum((gt == 1) & (pred == 0))

  def print(self):
    """Print the confusion matrix."""
    print("\t\t\tPredicted Positive\t|\tPredicted Negative")
    print("____________________________________________________________________")
    print(f"Actual Positive\t| {self.TP}\t\t| {self.FN}")
    print(f"Actual Negative\t| {self.FP}\t\t| {self.TN}\n")

  def sensitivity(self):
    """Calculate and return sensitivity (recall)."""
    return self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0

  def specificity(self):
    """Calculate and return specificity."""
    return self.TN / (self.TN + self.FP) if (self.TN + self.FP) > 0 else 0

  def dice(self):
    """Calculate and return the Dice coefficient."""
    return (2 * self.TP) / (2 * self.TP + self.FP + self.FN)