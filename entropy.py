import numpy as np
from scipy.stats import entropy as scientropy
from math import log, e
import pandas as pd

import timeit

# def entropy1(labels, base=None):
#   value,counts = np.unique(labels, return_counts=True)
#   return entropy(counts, base=base)

# def entropy(labels, base=None):
#   """ Computes entropy of label distribution. """

#   n_labels = len(labels)

#   if n_labels <= 1:
#     return 0

#   value,counts = np.unique(labels, return_counts=True)
#   probs = counts / n_labels
#   n_classes = np.count_nonzero(probs)

#   if n_classes <= 1:
#     return 0

#   ent = 0.

#   # Compute entropy
#   base = e if base is None else base
#   for i in probs:
#     ent -= i * log(i, base)

#   return ent

# def entropy3(labels, base=None):
#   vc = pd.Series(labels).value_counts(normalize=True, sort=False)
#   base = e if base is None else base
#   return -(vc * np.log(vc)/np.log(base)).sum()

def entropy(data, base=None):
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    print(counts)
    entropy = scientropy(counts)
    return entropy/255