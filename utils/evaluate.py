# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.ranking as rnk

def ideal_metrics(data_split, rank_weights, labels):
  cutoff = rank_weights.size
  result = np.zeros(data_split.num_queries())
  for qid in range(data_split.num_queries()):
    q_labels = data_split.query_values_from_vector(qid, labels)
    ranking = rnk.cutoff_ranking(-q_labels, cutoff)
    result[qid] = np.sum(rank_weights[:ranking.size]*q_labels[ranking])
  return result

def evaluate_max_likelihood(data_split, model, rank_weights, labels, ideal_metrics):
  cutoff = rank_weights.size
  scores = model(data_split.feature_matrix).numpy()[:, 0]
  
  result = 0.
  query_normalized_result = 0.
  for qid in range(data_split.num_queries()):
    q_scores = data_split.query_values_from_vector(qid, scores)
    q_labels = data_split.query_values_from_vector(qid, labels)
    ranking = rnk.cutoff_ranking(-q_scores, cutoff)
    q_result = np.sum(rank_weights[:ranking.size]*q_labels[ranking])
    result += q_result
    if ideal_metrics[qid] != 0:
      query_normalized_result += q_result/ideal_metrics[qid]
  result /= data_split.num_queries()
  query_normalized_result /= data_split.num_queries()
  normalized_result = result/np.mean(ideal_metrics)
  return result, normalized_result, query_normalized_result

def evaluate_expected(data_split, model, rank_weights, labels, ideal_metrics, num_samples):
  cutoff = rank_weights.size
  scores = model(data_split.feature_matrix).numpy()[:, 0]
  
  result = 0.
  query_normalized_result = 0.
  for qid in range(data_split.num_queries()):
    q_scores = data_split.query_values_from_vector(qid, scores)
    q_labels = data_split.query_values_from_vector(qid, labels)
    sampled_rankings = pl.gumbel_sample_rankings(
                                  q_scores,
                                  num_samples,
                                  cutoff=cutoff)[0]
    q_result = np.mean(
      np.sum(rank_weights[None,:sampled_rankings.shape[1]]*q_labels[sampled_rankings], axis=1)
      ,axis=0)
    result += q_result
    if ideal_metrics[qid] != 0:
      query_normalized_result += q_result/ideal_metrics[qid]
  result /= data_split.num_queries()
  query_normalized_result /= data_split.num_queries()
  normalized_result = result/np.mean(ideal_metrics)
  return result, normalized_result, query_normalized_result

def compute_results(data_split, model, rank_weights,
                    labels, ideal_metrics, num_samples):
  ML, N_ML, QN_ML = evaluate_max_likelihood(
                data_split, model, rank_weights,
                labels, ideal_metrics)
  E, N_E, QN_E = evaluate_expected(
                data_split, model, rank_weights,
                labels, ideal_metrics, num_samples)
  return {
      'maximum likelihood': ML,
      'normalized maximum likelihood': N_ML,
      'query normalized maximum likelihood': QN_ML,
      'expectation': E,
      'normalized expectation': N_E,
      'query normalized expectation': QN_E,
      }

def evaluate_fairness(data_split, model, rank_weights, labels, num_samples):
  cutoff = rank_weights.size
  scores = model(data_split.feature_matrix).numpy()[:, 0]
  
  result = 0.
  squared_result = 0.
  for qid in range(data_split.num_queries()):
    q_scores = data_split.query_values_from_vector(qid, scores)
    q_labels = data_split.query_values_from_vector(qid, labels)
    if np.sum(q_labels) > 0 and q_labels.size > 1:
      sampled_rankings = pl.gumbel_sample_rankings(
                                    q_scores,
                                    num_samples,
                                    cutoff=cutoff)[0]

      q_n_docs = q_labels.shape[0]
      q_cutoff = min(cutoff, q_n_docs)
      doc_exposure = np.zeros(q_n_docs, dtype=np.float64)
      np.add.at(doc_exposure, sampled_rankings, rank_weights[:q_cutoff])
      doc_exposure /= num_samples

      swap_reward = doc_exposure[:,None]*q_labels[None,:]

      q_result = np.mean((swap_reward-swap_reward.T)**2.)
      q_result *= q_n_docs/(q_n_docs-1.)

      q_squared = np.mean(np.abs(swap_reward-swap_reward.T))
      q_squared *= q_n_docs/(q_n_docs-1.)

      result += q_result
      squared_result += q_squared
  result /= data_split.num_queries()
  squared_result /= data_split.num_queries()
  return result, squared_result

def compute_fairness_results(data_split, model, rank_weights,
                             labels, num_samples):
  absolute, squared = evaluate_fairness(
                data_split, model, rank_weights,
                labels, num_samples)
  return {
      'expectation absolute': absolute,
      'expectation squared': squared,
      }