# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

from inception.slim import slim
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from coco_inception import get_inception_score_coco

import math
import os.path
import scipy.misc
# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir',
                           './inception_finetuned_models/birds_valid299/model.ckpt',
                           """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string('image_folder', 
							'../output/birds_stageII_kl/Model/netG_epoch_100/',
							"""Path where to load the images """)

tf.app.flags.DEFINE_integer('num_classes', 50,      # 20 for flowers
                            """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10,
                            """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 32, "batch size")
tf.app.flags.DEFINE_integer('gpu', 3, "The ID of GPU to use")
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


fullpath = FLAGS.image_folder
print(fullpath)


MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score_coco(images, splits=10):
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    print([n.name for n in tf.get_default_graph().as_graph_def().node][:10])
    # exit(0)
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, feed_dict={"ExpandDims:0": inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    # return np.mean(scores), np.std(scores)


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)


def get_inception_score(sess, images, pred_op):
    splits = FLAGS.splits
    # assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)

    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print(indices[i*bs + j])
            print(i*bs + j)
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        # print("%d of %d batches" % (i, n_batches))
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) -
              np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)


def load_data(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    # try:
                    #   print(img.shape[0])
                    # except:
                    #   print(filename)
                    images.append(img)
    print('images', len(images), images[0].shape)
    return images


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
              images,
              dropout_keep_prob=0.8,
              num_classes=num_classes,
              is_training=for_training,
              restore_logits=restore_logits,
              scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [FLAGS.batch_size, 299, 299, 3],
                    name='inputs')
                # print(inputs)

                logits, _ = inference(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = \
                    tf.slice(logits, [0, 1],
                             [FLAGS.batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, FLAGS.checkpoint_dir)
                print('Restore the model from %s).' % FLAGS.checkpoint_dir)
                images = load_data(fullpath)
                get_inception_score(sess, images, pred_op)

ext_mod = "bird"

# For COCO - uncomment below lines and comment abovw 2
if ext_mod != "coco":
  if __name__ == '__main__':
      tf.app.run()
else:
  images = load_data(fullpath)
  get_inception_score_coco(images)
