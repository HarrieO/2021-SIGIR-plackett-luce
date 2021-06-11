# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def lambdaloss(rank_weights,
               labels,
               scores,
               n_samples=None,
               sampled_inv_rankings=None,
               gumbel_scores=None):
  n_docs = scores.shape[0]
  cutoff = min(rank_weights.shape[0], n_docs)
  assert n_samples is not None or (sampled_inv_rankings is not None
                                    and gumbel_scores is not None)
  if sampled_inv_rankings is None:
    (_, sampled_inv_rankings, _, _,
    gumbel_scores) = pl.gumbel_sample_rankings(
                                  scores,
                                  n_samples,
                                  cutoff=None,
                                  inverted=True,
                                  return_gumbel=True)
  else:
    n_samples = sampled_inv_rankings.shape[0]
  
  (greater_i, lesser_i) = np.where(np.greater(
                            labels[:,None],
                            labels[None,:]))
  delta_rank = np.abs(sampled_inv_rankings[:, greater_i]
                      - sampled_inv_rankings[:, lesser_i])
  if n_docs > cutoff:
    safe_rank_weights = np.zeros(n_docs)
    safe_rank_weights[:cutoff] = rank_weights
  else:
    safe_rank_weights = rank_weights

  delta_weight = (safe_rank_weights[delta_rank-1]
                   - safe_rank_weights[delta_rank])
  pair_weight = delta_weight * (labels[None, greater_i]
                                - labels[None, lesser_i])

  exp_score_diff = np.exp(np.minimum(
              gumbel_scores[:,greater_i]
              - gumbel_scores[:,lesser_i],
              100))

  pair_deriv = pair_weight*exp_score_diff/(
               (exp_score_diff + 1.)*np.log(2.))
  pair_deriv = np.mean(pair_deriv, axis=0)

  doc_weights = np.zeros(n_docs, dtype=np.float64)
  np.add.at(doc_weights, greater_i, pair_deriv)
  np.add.at(doc_weights, lesser_i, -pair_deriv)

  return doc_weights

def lambdarank(rank_weights, labels, scores, n_samples):
  n_docs = scores.shape[0]
  cutoff = min(rank_weights.shape[0], n_docs)
  (_, sampled_inv_rankings, _, _,
    gumbel_scores) = pl.gumbel_sample_rankings(
                                  scores,
                                  n_samples,
                                  cutoff=None,
                                  inverted=True,
                                  return_gumbel=True)
  (greater_i, lesser_i) = np.where(np.greater(
                            labels[:,None],
                            labels[None,:]))
  delta_rank = np.abs(sampled_inv_rankings[:, greater_i]
                      - sampled_inv_rankings[:, lesser_i])
  if n_docs > cutoff:
    safe_rank_weights = np.zeros(n_docs)
    safe_rank_weights[:cutoff] = rank_weights
  else:
    safe_rank_weights = rank_weights

  delta_weight = np.mean(safe_rank_weights[delta_rank-1]
                   - safe_rank_weights[delta_rank], axis=0)
  pair_weight = delta_weight * (labels[greater_i]
                                - labels[lesser_i])
  exp_score_diff = np.exp(np.minimum(
              scores[greater_i] - scores[lesser_i],
              100))

  pair_deriv = pair_weight*exp_score_diff/(
               (exp_score_diff + 1.)*np.log(2.))

  doc_weights = np.zeros(n_docs, dtype=np.float64)
  np.add.at(doc_weights, greater_i, pair_deriv)
  np.add.at(doc_weights, lesser_i, -pair_deriv)

  return doc_weights