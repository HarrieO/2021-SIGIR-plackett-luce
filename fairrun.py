# Copyright (C) H.R. Oosterhuis 2021.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import tensorflow as tf
import json

import algorithms.PLRank as plr
import algorithms.pairwise as pw
import algorithms.lambdaloss as ll
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.nnmodel as nn
import utils.evaluate as evl
import utils.plackettluce as pl


parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--num_samples", required=True,
                    help="Number of samples for gradient estimation ('dynamic' applies the dynamic strategy).")
parser.add_argument("--num_eval_samples", type=int,
                    help="Number of samples for metric calculation in evaluation.",
                    default=10**2)
parser.add_argument("--num_exposure_samples", type=int,
                    help="Maximum number for estimating exposure.",
                    default=10**3)
parser.add_argument("--loss", type=str, required=True,
                    help="Name of the loss to use (PL_rank_1/PL_rank_2/lambdaloss/pairwise/policygradient/placementpolicygradient).")
parser.add_argument("--timed", action='store_true',
                    help="Turns off evaluation so method can be timed.")
parser.add_argument("--vali", action='store_true',
                    help="Results calculated on the validation set.")

args = parser.parse_args()

cutoff = args.cutoff
num_samples = args.num_samples
num_eval_samples = args.num_eval_samples
num_exposure_samples = args.num_exposure_samples
timed_run = args.timed
validation_results = args.vali

if num_samples == 'dynamic':
  dynamic_samples = True
else:
  dynamic_samples = False
  num_samples = int(num_samples)
  assert num_samples <= num_exposure_samples

if timed_run:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 20
    max_time = 3600
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 20
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 40
    max_time = 100*60*2
  elif args.dataset == 'istella':
    n_epochs = 40
    max_time = 100*60
else:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 20
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 20
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 20
  elif args.dataset == 'istella':
    n_epochs = 20

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

max_ranking_size = np.min((cutoff, data.max_query_size()))

model_params = {'hidden units': [32, 32],
                'learning_rate': 0.01,}

model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=model_params['learning_rate'])

results = []

metric_weights = 1./np.log2(np.arange(max_ranking_size) + 2)
train_labels = 2**data.train.label_vector-1
vali_labels = 2**data.validation.label_vector-1
test_labels = 2**data.test.label_vector-1

train_ndoc_feat = np.zeros(data.train.num_docs())
vali_ndoc_feat = np.zeros(data.validation.num_docs())
test_ndoc_feat = np.zeros(data.test.num_docs())
for qid in range(data.train.num_queries()):
  q_feat = data.train.query_values_from_vector(qid, train_ndoc_feat)
  q_feat[:] = q_feat.shape[0]
for qid in range(data.validation.num_queries()):
  q_feat = data.validation.query_values_from_vector(qid, vali_ndoc_feat)
  q_feat[:] = q_feat.shape[0]
for qid in range(data.test.num_queries()):
  q_feat = data.test.query_values_from_vector(qid, test_ndoc_feat)
  q_feat[:] = q_feat.shape[0]


data.train.feature_matrix = np.concatenate([data.train.feature_matrix,
                                            train_ndoc_feat[:,None]], axis=1)
data.validation.feature_matrix = np.concatenate([data.validation.feature_matrix,
                                                  vali_ndoc_feat[:,None]], axis=1)
data.test.feature_matrix = np.concatenate([data.test.feature_matrix,
                                          test_ndoc_feat[:,None]], axis=1)
data.num_features += 1

real_start_time = time.time()
total_train_time = 0
last_total_train_time = time.time()
method_train_time = 0

n_queries = data.train.num_queries()
if dynamic_samples:
  num_samples = 10
  float_num_samples = 10.
  add_per_step = 90./(n_queries*40.)
  max_num_samples = 100
if timed_run:
  step_between_check = (round(n_queries/100.)*100.)/10000.*n_epochs
else:
  step_between_check = (round(n_queries/100.)*100.)/100.*n_epochs

steps = 0
next_check = 0
check_i = 0
for epoch_i in range(n_epochs):
  query_permutation = np.random.permutation(n_queries)
  for qid in query_permutation:
    q_labels =  data.train.query_values_from_vector(
                              qid, train_labels)
    q_feat = data.train.query_feat(qid)

    if np.sum(q_labels) > 0 and q_labels.size > 1:
      q_n_docs = q_labels.shape[0]
      q_cutoff = min(cutoff, q_n_docs)
      q_metric_weights = metric_weights[:q_cutoff] #/q_ideal_metric
      with tf.GradientTape() as tape:
        q_tf_scores = model(q_feat)

        q_np_scores = q_tf_scores.numpy()[:,0]
        if args.loss == 'lambdaloss':
          (sampled_rankings, sampled_inv_rankings, _, _,
          gumbel_scores) = pl.gumbel_sample_rankings(
                                  q_np_scores,
                                  num_exposure_samples,
                                  cutoff=None,
                                  inverted=True,
                                  return_gumbel=True)
          sampled_rankings = sampled_rankings[:,:cutoff]
        else:
          sampled_rankings = pl.gumbel_sample_rankings(
                                        q_np_scores,
                                        num_exposure_samples,
                                        cutoff=q_cutoff)[0]

        doc_exposure = np.zeros(q_n_docs, dtype=np.float64)
        np.add.at(doc_exposure, sampled_rankings[:,1:], q_metric_weights[1:])
        doc_exposure /= num_exposure_samples

        max_score = np.amax(q_np_scores)
        first_prob = np.exp(q_np_scores-max_score)/np.sum(np.exp(q_np_scores-max_score))
        doc_exposure += first_prob*q_metric_weights[0]

        swap_reward = doc_exposure[:,None]*q_labels[None,:]
        pair_error = (swap_reward - swap_reward.T)
        q_eps = np.mean(pair_error*q_labels[:, None], axis=0)
        q_eps *= 4./(q_n_docs-1.)

        last_method_train_time = time.time()
        if args.loss == 'policygradient':
          loss = tfl.policy_gradient(
                                    q_metric_weights,
                                    q_eps,
                                    q_tf_scores,
                                    sampled_rankings=sampled_rankings[:num_samples,:]
                                    )
          method_train_time += time.time() - last_method_train_time
        elif args.loss == 'placementpolicygradient':
          loss = tfl.placement_policy_gradient(
                                    q_metric_weights,
                                    q_eps,
                                    q_tf_scores,
                                    sampled_rankings=sampled_rankings[:num_samples,:]
                                    )
          method_train_time += time.time() - last_method_train_time
        else:
          q_np_scores = q_tf_scores.numpy()[:,0]
          if args.loss == 'pairwise':
            doc_weights = pw.pairwise(q_eps,
                                      q_np_scores,
                                      )
          elif args.loss == 'lambdaloss':
            doc_weights = ll.lambdaloss(
                                      q_metric_weights,
                                      q_eps,
                                      q_np_scores,
                                      sampled_inv_rankings=sampled_inv_rankings[:num_samples,:],
                                      gumbel_scores=gumbel_scores[:num_samples,:],
                                      )
          elif args.loss == 'PL_rank_1':
            doc_weights = plr.PL_rank_1(
                                      q_metric_weights,
                                      q_eps,
                                      q_np_scores,
                                      sampled_rankings=sampled_rankings[:num_samples,:])
          elif args.loss == 'PL_rank_2':
            doc_weights = plr.PL_rank_2(
                                      q_metric_weights,
                                      q_eps,
                                      q_np_scores,
                                      sampled_rankings=sampled_rankings[:num_samples,:])
          else:
            raise NotImplementedError('Unknown loss %s' % args.loss)
          method_train_time += time.time() - last_method_train_time

          loss = -tf.reduce_sum(q_tf_scores[:,0] * doc_weights)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    steps += 1
    if dynamic_samples:
      float_num_samples = 10 + steps*90./(n_queries*40.)
      num_samples = min(int(np.round(float_num_samples)), max_num_samples)
    cur_epoch = steps/float(n_queries)
    if timed_run and cur_epoch >= next_check:
      total_train_time += time.time() - last_total_train_time
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'num_samples': num_samples})
      print('%0.02f method-time: %s total-time: %s' % (cur_epoch,
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      check_i += 1
      next_check = (check_i/10000.)*n_epochs
      last_total_train_time = time.time()
      if total_train_time >= max_time:
        break
    elif cur_epoch >= next_check:
      total_train_time += time.time() - last_total_train_time
      if validation_results:
        cur_result = evl.compute_fairness_results(data.validation,
                                    model, metric_weights,
                                    vali_labels,
                                    num_eval_samples)
      else:
        cur_result = evl.compute_fairness_results(data.test,
                                    model, metric_weights,
                                    test_labels,
                                    num_eval_samples)
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'result': cur_result,
                      'num_samples': num_samples})
      print('%0.02f absolute error: %s squared error: %s method-time: %s total-time: %s' % (cur_epoch,
            cur_result['expectation absolute'],
            cur_result['expectation squared'],
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      check_i += 1
      next_check = (check_i/100.)*n_epochs
      last_total_train_time = time.time()

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'run name': args.loss.replace('_', ' '),
  'loss': args.loss.replace('_', ' '),
  'model hyperparameters': model_params,
  'results': results,
  'number of samples': num_samples,
  'number of evaluation samples': num_eval_samples,
  'cutoff': cutoff,
}
if dynamic_samples:
  output['number of samples'] = 'dynamic'

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
