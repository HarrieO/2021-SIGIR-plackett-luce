# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import tensorflow as tf
import numpy as np
import utils.plackettluce as pl

def placement_policy_gradient(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  np_scores = scores.numpy()[:,0]
  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    np_scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)

  ninf_mask = np.zeros((n_samples, cutoff, n_docs), dtype=bool)
  ninf_mask[srange[:,None],
            crange[None,1:],
            sampled_rankings[:,:-1]] = True
  ninf_mask[:,:] = np.cumsum(ninf_mask, axis=1)

  sampled_scores = tf.gather(scores, sampled_rankings)[:,:,0]
  tiled_scores = tf.tile(scores[None,None,:,0], (n_samples, cutoff, 1))
  tiled_scores = tf.where(ninf_mask, np.NINF, tiled_scores)
  max_per_rank = np.max(tiled_scores, axis=2)
  tiled_scores -= max_per_rank[:,:,None]

  sample_denom = tf.reduce_logsumexp(tiled_scores, axis=2)
  sample_log_prob = sampled_scores-sample_denom

  rewards = rank_weights[None,:cutoff]*labels[sampled_rankings]
  cum_rewards = tf.cumsum(rewards, axis=1, reverse=True)
  
  result = tf.reduce_sum(tf.reduce_mean(
                  sample_log_prob*cum_rewards, axis=0))
  return -result

def policy_gradient(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  np_scores = scores.numpy()[:,0]
  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = pl.gumbel_sample_rankings(
                                    np_scores,
                                    n_samples,
                                    cutoff=cutoff)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  srange = np.arange(n_samples)
  crange = np.arange(cutoff)
  
  ninf_mask = np.zeros((n_samples, cutoff, n_docs), dtype=bool)
  ninf_mask[srange[:,None],
            crange[None,1:],
            sampled_rankings[:,:-1]] = True
  ninf_mask[:,:] = np.cumsum(ninf_mask, axis=1)

  sampled_scores = tf.gather(scores, sampled_rankings)[:,:,0]
  tiled_scores = tf.tile(scores[None,None,:,0], (n_samples, cutoff, 1))
  tiled_scores = tf.where(ninf_mask, np.NINF, tiled_scores)
  max_per_rank = np.max(tiled_scores, axis=2)
  tiled_scores -= max_per_rank[:,:,None]

  sample_denom = tf.reduce_logsumexp(tiled_scores, axis=2)
  sample_log_prob = sampled_scores-sample_denom
  final_prob_loss = tf.reduce_sum(sample_log_prob, axis=1)

  # print(sample_cum_prob[0,:].numpy())
  rewards = np.sum(rank_weights[None,:cutoff]
                   *labels[sampled_rankings], axis=1)
  result = tf.reduce_mean(
                  final_prob_loss*rewards, axis=0)
  return -result