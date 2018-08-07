from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#------------------------------------------------------------------------------------------------

all_models = ['DistNet', 'LocalNet']#, 'DeepBind', 'StandardNet']
dropout_status = [True]#, True, 	False, 	False, 	False, True,  True,  False]
l2_status = 	 [True]#, False, 	True, 	False, 	False, True,  False, True]
bn_status = 	 [True]#, False, 	False, 	True, 	False, False, True,  True]

# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')

# dataset path
data_path = '../data/Synthetic_dataset.h5'

# load dataset
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None


for i in range(len(dropout_status)):

	# loop through models
	for model_name in all_models:
		tf.reset_default_graph()
		print('model: ' + model_name)

		# compile neural trainer
		name = model_name
		if dropout_status[i]:
			name += '_do'
		if l2_status[i]:
			name += '_l2'
		if bn_status[i]:
			name += '_bn'
		model_path = utils.make_directory(params_path, model_name)
		file_path = os.path.join(model_path, name)

		# load model parameters
		model_layers, optimization, _ = helper.load_model(model_name, 
														  input_shape,
														  dropout_status[i], 
														  l2_status[i], 
														  bn_status[i])

		# build neural network class
		nnmodel = nn.NeuralNet(seed=247)
		nnmodel.build_layers(model_layers, optimization, supervised=True)

		nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

		# initialize session
		sess = utils.initialize_session()

		# set data in dictionary
		data = {'train': train, 'valid': valid, 'test': test}
		fit.train_minibatch(sess, nntrainer, data, batch_size=100, num_epochs=100, 
							patience=100, verbose=2, shuffle=True, save_all=False)	