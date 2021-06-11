# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

def pairwise(labels, scores):

  (greater, lesser) = np.where(
    np.greater(labels[:,None], labels[None,:]))

  max_pairs = np.maximum(scores[greater], scores[lesser])
  safe_greater = scores[greater] - max_pairs
  safe_lesser = scores[lesser] - max_pairs
  log_denom = np.log(np.exp(safe_greater)
                     + np.exp(safe_lesser))

  pair_weights = np.exp(
    safe_greater + safe_lesser - 2.*log_denom)

  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  np.add.at(result, greater, pair_weights)
  np.add.at(result, lesser, -pair_weights)

  return result
