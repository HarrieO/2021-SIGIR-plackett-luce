# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.ranking as rnk

def sample_rankings(log_scores, n_samples, cutoff=None, prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings = np.empty((n_samples, ranking_len), dtype=np.int32)
  inv_rankings = np.empty((n_samples, n_docs), dtype=np.int32)
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)

  if cutoff:
    inv_rankings[:] = ranking_len

  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    cumprobs = np.cumsum(probs, axis=1)
    random_values = np.random.uniform(size=n_samples)
    greater_equal_mask = np.greater_equal(random_values[:,None], cumprobs)
    sampled_ind = np.sum(greater_equal_mask, axis=1)

    rankings[:, i] = sampled_ind
    inv_rankings[ind, sampled_ind] = i
    rankings_prob[:, i] = probs[ind, sampled_ind]
    log_scores[ind, sampled_ind] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob

def gumbel_sample_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False, return_gumbel=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
  gumbel_scores = -(log_scores[None,:]+gumbel_samples)

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                gumbel_scores,
                                ranking_len,
                                invert=inverted)

  if not doc_prob:
    if not return_gumbel:
      return rankings, inv_rankings, None, None, None
    else:
      return rankings, inv_rankings, None, None, gumbel_scores

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    rankings_prob[:, i] = probs[ind, rankings[:, i]]
    log_scores[ind, rankings[:, i]] = np.NINF

  if return_gumbel:
    gumbel_return_values = gumbel_scores
  else:
    gumbel_return_values = None

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix, gumbel_return_values
  else:
    return rankings, inv_rankings, rankings_prob, None, gumbel_return_values

def metrics_based_on_samples(sampled_rankings,
                             weight_per_rank,
                             addition_per_rank,
                             weight_per_doc,):
  cutoff = sampled_rankings.shape[1]
  return np.sum(np.mean(
              weight_per_doc[sampled_rankings]*weight_per_rank[None, :cutoff],
            axis=0) + addition_per_rank[:cutoff], axis=0)

def datasplit_metrics(data_split,
                      policy_scores,
                      weight_per_rank,
                      addition_per_rank,
                      weight_per_doc,
                      query_norm_factors=None,
                      n_samples=1000):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)
  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    if not np.all(np.equal(q_doc_weights, 0.)):
      q_policy_scores = data_split.query_values_from_vector(qid, policy_scores)
      sampled_rankings = gumbel_sample_rankings(q_policy_scores,
                                                n_samples,
                                                cutoff=cutoff)[0]
      results[qid] = metrics_based_on_samples(sampled_rankings,
                                              weight_per_rank,
                                              addition_per_rank,
                                              q_doc_weights[:, None])
  if query_norm_factors is not None:
    results /= query_norm_factors

  return np.mean(results, axis=0)
