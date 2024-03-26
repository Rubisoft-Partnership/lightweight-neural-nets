"""
=========================================================
The Digit Dataset
=========================================================

This dataset is made up of 1797 8x8 images. Each image,
like the one shown below, is of a hand-written digit.
In order to utilize an 8x8 figure like this, we'd have to
first transform it into a feature vector with length 64.

See `here
<https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>`_
for more information about this dataset.

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# import matplotlib.pyplot as plt

from sklearn import datasets

# Load the digits dataset
(digits, targets) = datasets.load_digits(return_X_y=True)

with open("digits.txt", "w") as f:
    for (digit, target) in zip(digits, targets):
        # Convert the target into one-hot encoding
        one_hot = [0] * 10
        one_hot[target] = 1
        # Map digit to 0-1 range
        digit /= 16
        # Add padding for label embedding
        padding = " 0 0 0 0 0 0 0 0 0 0"
        # Write to the file space separated features and one-hot encoding
        f.write(" ".join(map(str, digit)) + padding + " " + " ".join(map(str, one_hot)) + "\n")
print("Done")
