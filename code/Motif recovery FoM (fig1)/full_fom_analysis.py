from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics
import helper
sys.path.append('../..')
import mutagenesisfunctions as mf

#-----------------------------------------------------------------------------------------------

def fom_heatmap(X, layer, alphabet, nntrainer, sess, eps=1e-7):
    
    def mutate(sequence, seq_length, dims):
        num_mutations = seq_length * dims
        hotplot_mutations = np.zeros((num_mutations,seq_length,1,dims)) 

        for position in range(seq_length):
            for nuc in range(dims):
                mut_seq = np.copy(sequence)          
                mut_seq[0, position, 0, :] = np.zeros(dims)
                mut_seq[0, position, 0, nuc] = 1.0

                hotplot_mutations[(position*dims)+nuc] = mut_seq
        return hotplot_mutations

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T
    
    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    #norm_heat_mut = heat_mut - predictions[0] + eps
    #norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    return (heat_mut)
#-----------------------------------------------------------------------------------------------

'''
This script outputs the First Order Mutagenesis scores for every model and every regularization mode over every positively labelled sequence.
It outputs a single np.array with the shape (num_models, num_reg, num_pos, dims, seqlen)

It will also output an array in the shape (num_models, num_reg) showing the name of each model for ease of indexing into the larger model.

'''
#------------------------------------------------------------------------------------------------

all_models = ['DistNet', 'LocalNet', 'DeepBind', 'StandardNet']
num_models = len(all_models) # number of models
dropout_status = [True, True, 	False, 	False, 	False, True,  True,  False]
l2_status = 	 [True, False, 	True, 	False, 	False, True,  False, True]
bn_status = 	 [True, False, 	False, 	True, 	False, False, True,  True]
num_reg = len(dropout_status) # number of regularization types

# save path
results_path = '../results'
params_path = utils.make_directory(results_path, 'model_params')

# dataset path
data_path = '../data/Synthetic_dataset.h5'

# load dataset
train, valid, test = helper.load_synthetic_dataset(data_path)
#Get the indices of correctly labelled sequences
right_index = np.where(test['targets'][:,0]==1)[0]
num_pos = len(right_index) # number of positively labelled sequences

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None

_, seqlen, _, dims = test['inputs'].shape # the length of each sequence and the number of dimensions

#initialize an array to hold the FoM results
full_fom_predictions = np.zeros((num_models, num_reg, num_pos, dims, seqlen))

# loop through models
for m, model_name in enumerate(all_models):

    #loop through every regularization type
    for r in range(len(dropout_status)):
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

        # compile neural trainers
        nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

        # initialize session
        sess = utils.initialize_session()

        # load best parameters
        nntrainer.set_best_parameters(sess)
        
        for p, index in enumerate(right_index):
            #Extract a sequence and perform First Order Mutagenesis
            X = np.expand_dims(test['inputs'][index], axis=0)
            
            full_fom_predictions[m, r, p, :, :] = fom_heatmap(X, 'output', 'dna', nntrainer, sess)

#save the array
import h5py
save_path = utils.make_directory(results_path, 'FoM_full_analysis' + '.hdf5')
hdf5path = os.path.join(save_path, 'full_fom_predictions')
with h5py.File(hdf5path, 'w') as f:
    f.create_dataset('full_fom', data=full_fom_predictions.astype(np.float32), compression='gzip')

        
        
        
        
