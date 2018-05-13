# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model defination for the RetinaNet Model.

Defines model_fn of RetinaNet for TF Estimator. The model_fn includes RetinaNet
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import anchors
import coco_metric
import retinanet_architecture
# from tensorflow.contrib.tpu.python.tpu import bfloat16
# from tensorflow.contrib.tpu.python.tpu import tpu_estimator
# from tensorflow.contrib.tpu.python.tpu import tpu_optimizer




# A collection of Learning Rate schecules:
# third_party/tensorflow_models/object_detection/utils/learning_schedules.py
def _learning_rate_schedule(base_learning_rate, lr_warmup_init, lr_warmup_step,
                            lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  lr_warmup_remainder = 1.0 - lr_warmup_init
  linear_warmup = [
      (lr_warmup_init + lr_warmup_remainder * (float(step) / lr_warmup_step),
       step) for step in range(0, lr_warmup_step, max(1, lr_warmup_step // 100))
  ]
  lr_schedule = linear_warmup + [[1.0, lr_warmup_step], [0.1, lr_drop_step]]
  learning_rate = base_learning_rate
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             base_learning_rate * mult)
  return learning_rate


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-alpha)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    total_loss = tf.reduce_sum(weighted_loss)
    total_loss /= normalizer
  return total_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  """Computes classification loss."""
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def _detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in
      [batch_size, height, width, num_anchors * num_classes].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integar tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integar tensor representing total class loss.
    box_loss: an integar tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
    cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                      [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]
    cls_losses.append(
        _classification_loss(
            cls_outputs[level],
            cls_targets_at_level,
            num_positives_sum,
            alpha=params['alpha'],
            gamma=params['gamma']))
    box_losses.append(
        _box_loss(
            box_outputs[level],
            box_targets_at_level,
            num_positives_sum,
            delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss


import collections
import copy
import signal
import threading
import time
import traceback

import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.summary import summary_ops as contrib_summary
from tensorflow.contrib.tpu.python.ops import tpu_ops
# from tensorflow.contrib.tpu.python.tpu import tpu
# from tensorflow.contrib.tpu.python.tpu import tpu_config
# from tensorflow.contrib.tpu.python.tpu import tpu_context
# from tensorflow.contrib.tpu.python.tpu import tpu_feed
# from tensorflow.contrib.tpu.python.tpu import training_loop
# from tensorflow.contrib.tpu.python.tpu import util as util_lib
from tensorflow.contrib.training.python.training import hparam
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


class _OutfeedHostCall(object):
  """Support for `eval_metrics` and `host_call` in TPUEstimatorSpec."""

  def __init__(self, ctx):
    self._ctx = ctx
    self._names = []
    # All of these are dictionaries of lists keyed on the name.
    self._host_fns = {}
    self._tensor_keys = collections.defaultdict(list)
    self._tensors = collections.defaultdict(list)
    self._tensor_dtypes = collections.defaultdict(list)
    self._tensor_shapes = collections.defaultdict(list)

  @staticmethod
  def validate(host_calls):
    """Validates the `eval_metrics` and `host_call` in `TPUEstimatorSpec`."""

    for name, host_call in host_calls.items():
      if not isinstance(host_call, (tuple, list)):
        raise ValueError('{} should be tuple or list'.format(name))
      if len(host_call) != 2:
        raise ValueError('{} should have two elements.'.format(name))
      if not callable(host_call[0]):
        raise TypeError('{}[0] should be callable.'.format(name))
      if not isinstance(host_call[1], (tuple, list, dict)):
        raise ValueError('{}[1] should be tuple or list, or dict.'.format(name))

      if isinstance(host_call[1], (tuple, list)):
        fullargspec = tf_inspect.getfullargspec(host_call[0])
        fn_args = util.fn_args(host_call[0])
        # wrapped_hostcall_with_global_step uses varargs, so we allow that.
        if fullargspec.varargs is None and len(host_call[1]) != len(fn_args):
          raise RuntimeError(
              'In TPUEstimatorSpec.{}, length of tensors {} does not match '
              'method args of the function, which takes {}.'.format(
                  name, len(host_call[1]), len(fn_args)))

  @staticmethod
  def create_cpu_hostcall(host_calls):
    """Runs on the host_call on CPU instead of TPU when use_tpu=False."""

    _OutfeedHostCall.validate(host_calls)
    ret = {}
    for name, host_call in host_calls.items():
      host_fn, tensors = host_call
      if isinstance(tensors, (tuple, list)):
        ret[name] = host_fn(*tensors)
      else:
        # Must be dict.
        try:
          ret[name] = host_fn(**tensors)
        except TypeError as e:
          logging.warning(
              'Exception while calling %s: %s. It is likely the tensors '
              '(%s[1]) do not match the '
              'function\'s arguments', name, e, name)
          raise e
    return ret

  def record(self, host_calls):
    """Records the host_call structure."""

    for name, host_call in host_calls.items():
      host_fn, tensor_list_or_dict = host_call
      self._names.append(name)
      self._host_fns[name] = host_fn

      if isinstance(tensor_list_or_dict, dict):
        for (key, tensor) in six.iteritems(tensor_list_or_dict):
          self._tensor_keys[name].append(key)
          self._tensors[name].append(tensor)
          self._tensor_dtypes[name].append(tensor.dtype)
          self._tensor_shapes[name].append(tensor.shape)
      else:
        # List or tuple.
        self._tensor_keys[name] = None
        for tensor in tensor_list_or_dict:
          self._tensors[name].append(tensor)
          self._tensor_dtypes[name].append(tensor.dtype)
          self._tensor_shapes[name].append(tensor.shape)

  def create_enqueue_op(self):
    """Create the op to enqueue the recorded host_calls.

    Returns:
      A list of enqueue ops, which is empty if there are no host calls.
    """
    if not self._names:
      return []

    tensors = []
    # TODO(jhseu): Consider deduping tensors.
    for name in self._names:
      tensors.extend(self._tensors[name])

    print("WARNING: not implemented error.")
    # with ops.device(tpu.core(0)):
    #   return [tpu_ops.outfeed_enqueue_tuple(tensors)]

  def create_tpu_hostcall(self):
    """Sends the tensors through outfeed and runs the host_fn on CPU.

    The tensors are concatenated along dimension 0 to form a global tensor
    across all shards. The concatenated function is passed to the host_fn and
    executed on the first host.

    Returns:
      A dictionary mapping name to the return type of the host_call by that
      name.

    Raises:
      RuntimeError: If outfeed tensor is scalar.
    """
    if not self._names:
      return []

    ret = {}
    # For each i, dequeue_ops[i] is a list containing the tensors from all
    # shards. This list is concatenated later.
    dequeue_ops = []
    tensor_dtypes = []
    tensor_shapes = []
    for name in self._names:
      for _ in self._tensors[name]:
        dequeue_ops.append([])
      for dtype in self._tensor_dtypes[name]:
        tensor_dtypes.append(dtype)
      for shape in self._tensor_shapes[name]:
        tensor_shapes.append(shape)

    # Outfeed ops execute on each replica's first logical core. Note: we must
    # constraint it such that we have at most one outfeed dequeue and enqueue
    # per replica.
    tpu_device_placement_fn = self._ctx.tpu_device_placement_function
    for i in xrange(self._ctx.num_replicas):
      with ops.device(tpu_device_placement_fn(i)):
        outfeed_tensors = tpu_ops.outfeed_dequeue_tuple(
            dtypes=tensor_dtypes, shapes=tensor_shapes)
        for j, item in enumerate(outfeed_tensors):
          dequeue_ops[j].append(item)

    # Deconstruct dequeue ops.
    dequeue_ops_by_name = {}
    pos = 0
    for name in self._names:
      dequeue_ops_by_name[name] = dequeue_ops[pos:pos+len(self._tensors[name])]
      pos += len(self._tensors[name])

    # It is assumed evaluation always happens on single host TPU system. So,
    # place all ops on tpu host if possible.
    #
    # TODO(jhseu): Evaluate whether this is right for summaries.
    with ops.device(self._ctx.tpu_host_placement_function(core_id=0)):
      for name in self._names:
        dequeue_ops = dequeue_ops_by_name[name]
        for i, item in enumerate(dequeue_ops):
          if dequeue_ops[i][0].shape.ndims == 0:
            raise RuntimeError(
                'All tensors outfed from TPU should preserve batch size '
                'dimension, but got scalar {}'.format(dequeue_ops[i][0]))
          # TODO(xiejw): Allow users to specify the axis for batch size
          # dimension.
          dequeue_ops[i] = array_ops.concat(dequeue_ops[i], axis=0)

        if self._tensor_keys[name] is not None:
          # The user-provided eval_metrics[1] is a dict.
          dequeue_ops = dict(zip(self._tensor_keys[name], dequeue_ops))
          try:
            ret[name] = self._host_fns[name](**dequeue_ops)
          except TypeError as e:
            logging.warning(
                'Exception while calling %s: %s. It is likely the tensors '
                '(%s[1]) do not match the '
                'function\'s arguments', name, e, name)
            raise e
        else:
          ret[name] = self._host_fns[name](*dequeue_ops)

    return ret


class _OutfeedHostCallHook(session_run_hook.SessionRunHook):
  """Hook to run host calls when use_tpu=False."""

  def __init__(self, tensors):
    self._tensors = tensors

  def begin(self):
    # We duplicate this code from the TPUInfeedOutfeedSessionHook rather than
    # create a separate hook to guarantee execution order, because summaries
    # need to be initialized before the outfeed thread starts.
    # TODO(jhseu): Make a wrapper hook instead?
    self._init_ops = contrib_summary.summary_writer_initializer_op()
    # Get all the writer resources from the initializer, so we know what to
    # flush.
    self._finalize_ops = []
    for op in self._init_ops:
      self._finalize_ops.append(contrib_summary.flush(writer=op.inputs[0]))

  def after_create_session(self, session, coord):
    session.run(self._init_ops)

  def before_run(self, run_context):
    return basic_session_run_hooks.SessionRunArgs(self._tensors)

  def end(self, session):
    session.run(self._finalize_ops)



class ModifiedTPUEstimatorSpec(
    collections.namedtuple('TPUEstimatorSpec', [
        'mode',
        'predictions',
        'loss',
        'train_op',
        'eval_metrics',
        'export_outputs',
        'scaffold_fn',
        'host_call'
    ])):
  """Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  See `EstimatorSpec` for `mode`, 'predictions, 'loss', 'train_op', and
  'export_outputs`.

  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
  `metric_fn` runs on CPU to generate metrics and `tensors` represents the
  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
  To be precise, TPU evaluation expects a slightly different signature from the
  @{tf.estimator.Estimator}. While `EstimatorSpec.eval_metric_ops` expects a
  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
  `tensors` usually specify the model logits, which are transferred back from
  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
  size is the first dimension. Once all tensors are available at CPU host from
  all shards, they are concatenated (on CPU) and passed as positional arguments
  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
  dict. `metric_fn` takes the `tensors` and returns a dict from metric string
  name to the result of calling a metric function, namely a `(metric_tensor,
  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
  `eval_metrics`.

  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
  function should not capture any Tensors in `model_fn`.

  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`
  to pass to that function and returns a list of Tensors. `host_call` currently
  works for train() and evaluate(). The Tensors returned by the function is
  executed on the CPU on every step, so there is communication overhead when
  sending tensors from TPU to CPU. To reduce the overhead, try reducing the
  size of the tensors. The `tensors` are concatenated along their major (batch)
  dimension, and so must be >= rank 1. The `host_call` is useful for writing
  summaries with @{tf.contrib.summary.create_file_writer}.
  """

  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metrics=None,
              export_outputs=None,
              scaffold_fn=None,
              host_call=None):
    """Creates a validated `TPUEstimatorSpec` instance."""
    host_calls = {}
    if eval_metrics is not None:
      host_calls['eval_metrics'] = eval_metrics
    if host_call is not None:
      host_calls['host_call'] = host_call
    _OutfeedHostCall.validate(host_calls)
    return super(ModifiedTPUEstimatorSpec, cls).__new__(
        cls,
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        export_outputs=export_outputs,
        scaffold_fn=scaffold_fn,
        host_call=host_call)

  def as_estimator_spec(self):
    """Creates an equivalent `EstimatorSpec` used by CPU train/eval."""
    host_calls = {}
    if self.eval_metrics is not None:
      host_calls['eval_metrics'] = self.eval_metrics
    if self.host_call is not None:
      host_calls['host_call'] = self.host_call
    host_call_ret = _OutfeedHostCall.create_cpu_hostcall(host_calls)
    eval_metric_ops = None
    if self.eval_metrics is not None:
      eval_metric_ops = host_call_ret['eval_metrics']
    hooks = None
    if self.host_call is not None:
      hooks = [_OutfeedHostCallHook(host_call_ret['host_call'])]
    scaffold = self.scaffold_fn() if self.scaffold_fn else None
    return model_fn_lib.EstimatorSpec(
        mode=self.mode,
        predictions=self.predictions,
        loss=self.loss,
        train_op=self.train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=self.export_outputs,
        scaffold=scaffold,
        training_hooks=hooks,
        evaluation_hooks=hooks,
        prediction_hooks=hooks)



def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model defination for the RetinaNet model based on ResNet.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default bhn
      settings are in default_hparams function in this file.
    model: the RetinaNet model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
  """
  def _model_outputs():
    return model(
        features,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_classes=params['num_classes'],
        num_anchors=len(params['aspect_ratios'] * params['num_scales']),
        resnet_depth=params['resnet_depth'],
        is_training_bn=params['is_training_bn'])

  if params['use_bfloat16']:
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
    # with bfloat16.bfloat16_scope():
    #   cls_outputs, box_outputs = _model_outputs()
    #   levels = cls_outputs.keys()
    #   for level in levels:
    #     cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
    #     box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Load pretrained model from checkpoint.
  if params['resnet_checkpoint'] and mode == tf.estimator.ModeKeys.TRAIN:

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
          '/': 'resnet%s/' % params['resnet_depth'],
      })
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  # Set up training loss and learning rate.
  global_step = tf.train.get_global_step()
  learning_rate = _learning_rate_schedule(
      params['learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['lr_drop_step'], global_step)
  # cls_loss and box_loss are for logging. only total_loss is optimized.

  print("cls_outputs is", cls_outputs, "box_outputs is", box_outputs)
  total_loss, cls_loss, box_loss = _detection_loss(cls_outputs, box_outputs,
                                                   labels, params)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
        # we don't use TPU so we will just stick with the default optimizer
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=params['momentum'])
        # optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = variable_filter_fn(
        tf.trainable_variables(),
        params['resnet_depth']) if variable_filter_fn else None
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(total_loss, global_step, var_list=var_list)
  else:
    train_op = None

  # Evaluation only works on GPU/CPU host and batch_size=1
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(**kwargs):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
      
      print("params['aspect_ratios'] is", params['aspect_ratios'])
      print("params is", params)
      eval_anchors = anchors.Anchors(params['min_level'],
                                     params['max_level'],
                                     params['num_scales'],
                                     params['aspect_ratios'],
                                     params['anchor_scale'],
                                     params['image_size'])
      anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                             params['num_classes'])
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])
      # add metrics to output
      cls_outputs = {}
      box_outputs = {}
      for level in range(params['min_level'], params['max_level'] + 1):
        cls_outputs[level] = kwargs['cls_outputs_%d' % level]
        box_outputs[level] = kwargs['box_outputs_%d' % level]
        # print("cls_outputs[level]", cls_outputs[level],"box_outputs[level]", box_outputs[level], "level", level )
      detections = anchor_labeler.generate_detections(
          cls_outputs, box_outputs, kwargs['source_ids'])
      eval_metric = coco_metric.EvaluationMetric(params['val_json_file'])
      coco_metrics = eval_metric.estimator_metric_fn(detections,
                                                     kwargs['image_scales'])
      # Add metrics to output.
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    batch_size = params['batch_size']
    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            batch_size,
        ]), [batch_size, 1])
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': labels['source_ids'],
        'image_scales': labels['image_scales'],
    }
    for level in range(params['min_level'], params['max_level'] + 1):
      metric_fn_inputs['cls_outputs_%d' % level] = cls_outputs[level]
      metric_fn_inputs['box_outputs_%d' % level] = box_outputs[level]
    eval_metrics = (metric_fn, metric_fn_inputs)

  temp = ModifiedTPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)
  return temp.as_estimator_spec()

  # return ModifiedTPUEstimatorSpec(
  #     mode=mode,
  #     loss=total_loss,
  #     train_op=train_op,
  #     eval_metrics=eval_metrics,
  #     scaffold_fn=scaffold_fn)
  # return tpu_estimator.TPUEstimatorSpec(
  #     mode=mode,
  #     loss=total_loss,
  #     train_op=train_op,
  #     eval_metrics=eval_metrics,
  #     scaffold_fn=scaffold_fn)


def retinanet_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=retinanet_architecture.retinanet,
      variable_filter_fn=retinanet_architecture.remove_variables)


def default_hparams():
  return tf.contrib.training.HParams(
      image_size=640,
      input_rand_hflip=True,
      # dataset specific parameters
      num_classes=90,
      skip_crowd=True,
      # model architecture
      min_level=3,
      max_level=7,
      num_scales=3,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=4.0,
      resnet_depth=50,
      # is batchnorm training mode
      is_training_bn=True,
      # optimization
      momentum=0.9,
      learning_rate=0.08,
      lr_warmup_init=0.1,
      lr_warmup_step=2000,
      lr_drop_step=15000,
      # classification loss
      alpha=0.25,
      gamma=1.5,
      # localization loss
      delta=0.1,
      box_loss_weight=50.0,
      # resnet checkpoint
      resnet_checkpoint=None,
      # output detection
      box_max_detected=100,
      box_iou_threshold=0.5,
      use_bfloat16=False,
  )
