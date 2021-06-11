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
timed_run = args.timed
validation_results = args.vali

if num_samples == 'dynamic':
  dynamic_samples = True
else:
  dynamic_samples = False
  num_samples = int(num_samples)

if timed_run:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 40
    max_time = 8000
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 40
    max_time = 9000
  elif args.dataset == 'istella':
    n_epochs = 40
    max_time = 15000
else:
  if args.dataset == 'Webscope_C14_Set1':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB10k':
    n_epochs = 40
  elif args.dataset == 'MSLR-WEB30k':
    n_epochs = 40
  elif args.dataset == 'istella':
    n_epochs = 40

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
                'learning_rate': 0.001,}

model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=model_params['learning_rate'])

results = []

metric_weights = 1./np.log2(np.arange(max_ranking_size) + 2)
train_labels = 2**data.train.label_vector-1
vali_labels = 2**data.validation.label_vector-1
test_labels = 2**data.test.label_vector-1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_vali_metrics = evl.ideal_metrics(data.validation, metric_weights, vali_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

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
steps = 0
next_check = 0
for epoch_i in range(n_epochs):
  query_permutation = np.random.permutation(n_queries)
  for qid in query_permutation:
    q_labels =  data.train.query_values_from_vector(
                              qid, train_labels)
    q_feat = data.train.query_feat(qid)
    q_ideal_metric = ideal_train_metrics[qid]

    if q_ideal_metric != 0:
      q_metric_weights = metric_weights #/q_ideal_metric #uncomment for NDCG
      with tf.GradientTape() as tape:
        q_tf_scores = model(q_feat)

        last_method_train_time = time.time()
        if args.loss == 'policygradient':
          loss = tfl.policy_gradient(
                                    q_metric_weights,
                                    q_labels,
                                    q_tf_scores,
                                    n_samples=num_samples
                                    )
          method_train_time += time.time() - last_method_train_time
        elif args.loss == 'placementpolicygradient':
          loss = tfl.placement_policy_gradient(
                                    q_metric_weights,
                                    q_labels,
                                    q_tf_scores,
                                    n_samples=num_samples
                                    )
          method_train_time += time.time() - last_method_train_time
        else:
          q_np_scores = q_tf_scores.numpy()[:,0]
          if args.loss == 'pairwise':
            doc_weights = pw.pairwise(q_labels,
                                      q_np_scores,
                                      )
          elif args.loss == 'lambdaloss':
            doc_weights = ll.lambdaloss(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples
                                      )
          elif args.loss == 'PL_rank_1':
            doc_weights = plr.PL_rank_1(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples)
          elif args.loss == 'PL_rank_2':
            doc_weights = plr.PL_rank_2(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples)
          else:
            raise NotImplementedError('Unknown loss %s' % args.loss)
          method_train_time += time.time() - last_method_train_time

          loss = -tf.reduce_sum(q_tf_scores[:,0] * doc_weights)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    steps += 1
    if dynamic_samples:
      float_num_samples = 10 + steps*add_per_step
      num_samples = min(int(np.round(float_num_samples)), max_num_samples)
    cur_epoch = steps/float(n_queries)
    if timed_run and (cur_epoch > next_check or (time.time() - real_start_time) > max_time):
      total_train_time += time.time() - last_total_train_time
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'num_samples': num_samples})
      print('%0.02f method-time: %s total-time: %s' % (cur_epoch,
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      next_check += n_epochs/10000.
      last_total_train_time = time.time()
    elif cur_epoch >= next_check:
      total_train_time += time.time() - last_total_train_time
      if validation_results:
        cur_result = evl.compute_results(data.validation,
                                    model, metric_weights,
                                    vali_labels, ideal_vali_metrics,
                                    num_eval_samples)
      else:
        cur_result = evl.compute_results(data.test,
                                    model, metric_weights,
                                    test_labels, ideal_test_metrics,
                                    num_eval_samples)  
      cur_epoch = steps/float(n_queries)
      results.append({'steps': steps,
                      'epoch': next_check,
                      'train time': method_train_time,
                      'total time': total_train_time,
                      'result': cur_result,
                      'num_samples': num_samples})
      print('%0.02f expected_metric: %s deterministic_metric: %s method-time: %s total-time: %s' % (cur_epoch,
            cur_result['normalized expectation'], cur_result['normalized maximum likelihood'],
            method_train_time/cur_epoch, total_train_time/cur_epoch))
      next_check += n_epochs/100.
      last_total_train_time = time.time()

    if timed_run and (time.time() - real_start_time) > max_time:
      break

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
