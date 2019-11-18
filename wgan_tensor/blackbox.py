# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
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

"""Testing blackbox Defense-GAN models. This module is based on MNIST tutorial
of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import os
import re
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf

from models.gan import MnistDefenseGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.visualize import save_images_files

FLAGS = tf.app.flags.FLAGS
dataset_gan_dict = {
	'mnist': MnistDefenseGAN
}


def blackbox(gan, rec_data_path=None, batch_size=128,
             learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
             nb_epochs_s=10, lmbda=0.1, online_training=False,
             train_on_recs=False, test_on_dev=True,
             defense_type='none'):
	"""MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697

	Args:
		train_start: index of first training set example
		train_end: index of last training set example
		test_start: index of first test set example
		test_end: index of last test set example
		defense_type: Type of defense against blackbox attacks

	Returns:
		a dictionary with:
			 * black-box model accuracy on test set
			 * substitute model accuracy on test set
			 * black-box model accuracy on adversarial examples transferred
			   from the substitute model
	"""
	FLAGS = flags.FLAGS

	# Dictionary used to keep track and return key accuracies.
	accuracies = {}

	# Create TF session.
	sess = gan.sess


	if FLAGS.debug:
		debug_dir = os.path.join('debug', 'blackbox', FLAGS.debug_dir)
		ensure_dir(debug_dir)
		x_debug_test = test_images[:batch_size]

	# Redefine test set as remaining samples unavailable to adversaries
	if FLAGS.num_tests > 0:
		test_images = test_images[:FLAGS.num_tests]
		test_labels = test_labels[:FLAGS.num_tests]

	test_images = test_images[holdout:]
	test_labels = test_labels[holdout:]

	x_shape, classes = list(train_images.shape[1:]), train_labels.shape[1]

	# Define input and output TF placeholders

	if FLAGS.image_dim[0] == 3:
		FLAGS.image_dim = [FLAGS.image_dim[1], FLAGS.image_dim[2],
		                   FLAGS.image_dim[0]]

	images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
	labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

	rng = np.random.RandomState([11, 24, 1990])
	tf.set_random_seed(11241990)

	accuracies['sub'] = 0
	# Initialize the Fast Gradient Sign Method (FGSM) attack object.

	# Craft adversarial examples using the substitute.
	eval_params = {'batch_size': batch_size}

	# todo: here x_adv_sub is epsilon image

	if FLAGS.debug and gan is not None:  # To see some qualitative results.
		reconstructed_tensors = gan.reconstruct(x_adv_sub, batch_size=batch_size,
		                                        reconstructor_id=2)

		x_rec_orig = gan.reconstruct(images_tensor, batch_size=batch_size,
		                             reconstructor_id=3)
		x_adv_sub_val = sess.run(x_adv_sub,
		                         feed_dict={images_tensor: x_debug_test,
		                                    K.learning_phase(): 0})
		sess.run(tf.local_variables_initializer())
		x_rec_debug_val, x_rec_orig_val = sess.run(
			[reconstructed_tensors, x_rec_orig],
			feed_dict={
				images_tensor: x_debug_test,
				K.learning_phase(): 0})

		save_images_files(x_adv_sub_val, output_dir=debug_dir,
		                  postfix='adv')

		postfix = 'gen_rec'
		save_images_files(x_rec_debug_val, output_dir=debug_dir,
		                  postfix=postfix)
		save_images_files(x_debug_test, output_dir=debug_dir,
		                  postfix='orig')
		save_images_files(x_rec_orig_val, output_dir=debug_dir,
		                  postfix='orig_rec')
		return

	reconstructed_tensors = gan.reconstruct(
		x_adv_sub, batch_size=batch_size, reconstructor_id=4,
	)

	num_dims = len(images_tensor.get_shape())
	avg_inds = list(range(1, num_dims))
	diff_op = tf.reduce_mean(tf.square(x_adv_sub - reconstructed_tensors),
	                         axis=avg_inds)

	outs = model_eval_gan(sess, images_tensor, labels_tensor,
	                      predictions=model(reconstructed_tensors),
	                      test_images=test_images, test_labels=test_labels,
	                      args=eval_params, diff_op=diff_op,
	                      feed={K.learning_phase(): 0})

	accuracies['bbox_on_sub_adv_ex'] = outs[0]
	accuracies['roc_info'] = outs[1]
	print('Test accuracy of oracle on adversarial examples generated '
	      'using the substitute: ' + str(outs[0]))

	return accuracies


def _get_results_dir_filename(gan):
	result_file_name = 'sub={:d}_eps={:.2f}.txt'.format(FLAGS.data_aug,
	                                                    FLAGS.fgsm_eps)

	results_dir = os.path.join('results', '{}_{}'.format(
		FLAGS.defense_type, FLAGS.dataset_name))

	if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':
		results_dir = gan.checkpoint_dir.replace('output', 'results')
		result_file_name = \
			'teRR={:d}_teLR={:.4f}_teIter={:d}_sub={:d}_eps={:.2f}.txt'.format(
				gan.rec_rr,
				gan.rec_lr,
				gan.rec_iters,
				FLAGS.data_aug,
				FLAGS.fgsm_eps)

		if not FLAGS.train_on_recs:
			result_file_name = 'orig_' + result_file_name
	elif FLAGS.defense_type == 'adv_tr':
		result_file_name = 'sub={:d}_trEps={:.2f}_eps={:.2f}.txt'.format(
			FLAGS.data_aug, FLAGS.fgsm_eps_tr,
			FLAGS.fgsm_eps)
	if FLAGS.num_tests > -1:
		result_file_name = 'numtest={}_'.format(
			FLAGS.num_tests) + result_file_name

	if FLAGS.num_train > -1:
		result_file_name = 'numtrain={}_'.format(
			FLAGS.num_train) + result_file_name

	result_file_name = 'bbModel={}_subModel={}_'.format(FLAGS.bb_model,
	                                                    FLAGS.sub_model) \
	                   + result_file_name
	return results_dir, result_file_name


def main(cfg, argv=None):
	FLAGS = tf.app.flags.FLAGS
	GAN = dataset_gan_dict[FLAGS.dataset_name]

	gan = GAN(cfg=cfg, test_mode=True)
	gan.load_generator()
	# Setting test time reconstruction hyper parameters.
	[tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
	if FLAGS.defense_type.lower() != 'none':
		if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':

			# extract hyper parameters from reconstruction path.
			if FLAGS.rec_path:
				train_param_re = re.compile('recs_rr(.*)_lr(.*)_iters(.*)')
				[tr_rr, tr_lr, tr_iters] = \
					train_param_re.findall(FLAGS.rec_path)[0]
				gan.rec_rr = int(tr_rr)
				gan.rec_lr = float(tr_lr)
				gan.rec_iters = int(tr_iters)
		elif FLAGS.defense_type == 'defense_gan':
			assert FLAGS.online_training or not FLAGS.train_on_recs

	if FLAGS.override:
		gan.rec_rr = int(tr_rr)
		gan.rec_lr = float(tr_lr)
		gan.rec_iters = int(tr_iters)

	# Setting the reuslts directory
	results_dir, result_file_name = _get_results_dir_filename(gan)

	# Result file name. The counter makes sure we are not overwriting the
	# results.
	counter = 0
	temp_fp = str(counter) + '_' + result_file_name
	results_dir = os.path.join(results_dir, FLAGS.results_dir)
	temp_final_fp = os.path.join(results_dir, temp_fp)
	while os.path.exists(temp_final_fp):
		counter += 1
		temp_fp = str(counter) + '_' + result_file_name
		temp_final_fp = os.path.join(results_dir, temp_fp)
	result_file_name = temp_fp
	sub_result_path = os.path.join(results_dir, result_file_name)

	accuracies = blackbox(gan, rec_data_path=FLAGS.rec_path,
	                      batch_size=FLAGS.batch_size,
	                      learning_rate=FLAGS.learning_rate,
	                      nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
	                      data_aug=FLAGS.data_aug,
	                      nb_epochs_s=FLAGS.nb_epochs_s,
	                      lmbda=FLAGS.lmbda,
	                      online_training=FLAGS.online_training,
	                      train_on_recs=FLAGS.train_on_recs,
	                      defense_type=FLAGS.defense_type)

	ensure_dir(results_dir)

	with open(sub_result_path, 'a') as f:
		f.writelines([str(accuracies[x]) + ' ' for x in
		              ['bbox', 'sub', 'bbox_on_sub_adv_ex']])
		f.write('\n')
		print('[*] saved accuracy in {}'.format(sub_result_path))

	if 'roc_info' in accuracies.keys():  # For attack detection.
		pkl_result_path = sub_result_path.replace('.txt', '_roc.pkl')
		with open(pkl_result_path, 'w') as f:
			pickle.dump(accuracies['roc_info'], f, pickle.HIGHEST_PROTOCOL)
			print('[*] saved roc_info in {}'.format(sub_result_path))


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--cfg', required=True, help='Config file')

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args, _ = parser.parse_known_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	# Note: The load_config() call will convert all the parameters that are defined in
	# experiments/config files into FLAGS.param_name and can be passed in from command line.
	# arguments : python blackbox.py --cfg <config_path> --<param_name> <param_value>
	cfg = load_config(args.cfg)
	flags = tf.app.flags

	flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
	flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training '
	                                           'the black-box model.')
	flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train the '
	                                      'blackbox model.')
	flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary.')
	flags.DEFINE_integer('data_aug', 6, 'Number of substitute data augmentations.')
	flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute.')
	flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
	flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
	flags.DEFINE_float('fgsm_eps_tr', 0.15, 'FGSM epsilon for adversarial '
	                                        'training.')
	flags.DEFINE_string('rec_path', None, 'Path to Defense-GAN '
	                                      'reconstructions.')
	flags.DEFINE_integer('num_tests', 2000, 'Number of test samples.')
	flags.DEFINE_integer('random_test_iter', -1,
	                     'Number of random sampling for testing the '
	                     'classifier.')
	flags.DEFINE_boolean("online_training", False,
	                     'Train the base classifier based on online '
	                     'reconstructions from Defense-GAN, as opposed to '
	                     'using the cached reconstructions.')
	flags.DEFINE_string("defense_type", "none", "Type of defense "
	                                            "[defense_gan|adv_tr|none]")
	flags.DEFINE_string("results_dir", None, "The path to results.")
	flags.DEFINE_boolean("train_on_recs", False,
	                     "Train the black-box model on Defense-GAN "
	                     "reconstructions.")
	flags.DEFINE_integer('num_train', -1, 'Number of training samples for '
	                                      'the black-box model.')
	flags.DEFINE_string("bb_model", 'F',
	                    "The architecture of the classifier model.")
	flags.DEFINE_string("sub_model", 'E', "The architecture of the "
	                                      "substitute model.")
	flags.DEFINE_string("debug_dir", None, "Directory for debug outputs.")
	flags.DEFINE_boolean("debug", None, "Directory for debug outputs.")
	flags.DEFINE_boolean("override", None, "Overrides the test hyperparams.")

	main_cfg = lambda x: main(cfg, x)
	tf.app.run(main=main_cfg)