# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def PL_rank_1(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)

  weighted_labels = labels[sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, sampled_rankings, cumsum_labels)
  result /= n_samples

  ninf_mask = np.zeros((n_samples, cutoff-1, n_docs), dtype=np.float64)
  ninf_mask[srange[:,None],
            crange[None,:-1],
            sampled_rankings[:,:-1]] = np.NINF
  ninf_mask[:,:] = np.cumsum(ninf_mask, axis=1)

  tiled_scores = np.tile(scores[None,None,:], (n_samples, cutoff, 1))
  tiled_scores[:,1:,:] += ninf_mask
  max_per_rank = np.max(tiled_scores, axis=2)
  tiled_scores -= max_per_rank[:,:,None]

  denom_per_rank = np.log(np.sum(np.exp(tiled_scores), axis=2))
  prob_per_rank = np.exp(tiled_scores - denom_per_rank[:,:,None])

  minus_weights = np.mean(
    np.sum(prob_per_rank*cumsum_labels[:,:,None], axis=1)
    , axis=0, dtype=np.float64)

  result -= minus_weights

  return result


def PL_rank_2(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)

  relevant_docs = np.where(np.not_equal(labels, 0))[0]
  n_relevant_docs = relevant_docs.size

  weighted_labels = labels[sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, sampled_rankings[:,:-1], cumsum_labels[:,1:])
  result /= n_samples

  ninf_mask = np.zeros((n_samples, cutoff-1, n_docs), dtype=np.float64)
  ninf_mask[srange[:,None],
            crange[None,:-1],
            sampled_rankings[:,:-1]] = np.NINF
  ninf_mask[:,:] = np.cumsum(ninf_mask, axis=1)

  tiled_scores = np.tile(scores[None,None,:], (n_samples, cutoff, 1))
  tiled_scores[:,1:,:] += ninf_mask
  max_per_rank = np.max(tiled_scores, axis=2)
  tiled_scores -= max_per_rank[:,:,None]

  denom_per_rank = np.log(np.sum(np.exp(tiled_scores), axis=2))
  prob_per_rank = np.exp(tiled_scores - denom_per_rank[:,:,None])

  result -= np.mean(
    np.sum(prob_per_rank*cumsum_labels[:,:,None], axis=1)
    , axis=0, dtype=np.float64)
  result[relevant_docs] += np.mean(
    np.sum(prob_per_rank[:,:,relevant_docs]*(
                          rank_weights[None,:cutoff,None]
                          *labels[None,None,relevant_docs]), axis=1)
    , axis=0, dtype=np.float64)

  return result

