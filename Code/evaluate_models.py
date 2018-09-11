import tensorflow as tf
import numpy as np
from Tensorflow.Code_PLUS.Dataset import Dataset
import Tensorflow.Code_PLUS.sample_plus as sample_plus
import argparse
from time import time
from Tensorflow.Code_PLUS.NeuMF_PLUS import get_neuMF_PLUS_model
from Tensorflow.Code_PLUS.MLP_PLUS import get_MLP_PLUS_model
from Tensorflow.Code_TF_Original.GMF import get_MF_model
from Tensorflow.Code_TF_Original.MLP import get_MLP_model
from Tensorflow.Code_TF_Original.NeuMF import get_neuMF_model
def parse_args():
	parser = argparse.ArgumentParser(description="Run PLUS models.")
	parser.add_argument('--dir_path', nargs='?', default='../Models/',
						help='Model save path')
	parser.add_argument('--path', nargs='?', default='../Data/exp_ml-20m_tmp/',
						help='Input data path.')
	parser.add_argument('--dataset', nargs='?', default='exp_ml-20m_tmp',
						help='Choose a dataset.')
	parser.add_argument('--epochs', type=int, default=50,
						help='Number of epochs.')
	parser.add_argument('--batch_size', type=int, default=128,
						help='Batch size.')
	parser.add_argument('--num_factors', type=int, default=32,
						help='Embedding size of MF model.')
	parser.add_argument('--layers', nargs='?', default='[256,128,64,32]',
						help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument('--reg_mf', type=float, default=0,
						help='Regularization for MF embeddings.')					
	parser.add_argument('--reg_layers', nargs='?', default='[0.01,0.01,0.01,0.01]',
						help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate.')
	parser.add_argument('--learner', nargs='?', default='adam',
						help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
	parser.add_argument('--num_train_neg', type=int, default=4,
						help='Number of train negatives')
	parser.add_argument('--num_test_neg', type=int, default=1000,
						help='Number of test negatives')
	parser.add_argument('--top_number', type=int, default=10,
						help='Top K number')
	parser.add_argument('--seed', type=int, default=0,
						help='Random Seed')
	return parser.parse_args()

def main(unused_argv):
	args = parse_args()
	model_save = args.dir_path
	num_epochs = args.epochs
	batch_size = args.batch_size
	mf_dim = args.num_factors
	layers = eval(args.layers)
	reg_layers = eval(args.reg_layers)
	learning_rate = args.lr
	learner = args.learner
	num_train_neg = args.num_train_neg
	num_test_neg = args.num_test_neg
	dataset = Dataset(args.path + args.dataset)
	feature_arr, train, testRatings = dataset.feature_arr, dataset.trainMatrix, dataset.testRatings
	num_users, num_items = train.shape
	seed = args.seed

	params = {
	'num_users' : num_users,
	'num_items' : num_items,
	'mf_dim' : mf_dim,
	'layers' : layers,
	'reg_layers' : reg_layers,
	'learning_rate' : learning_rate,
	'learner' : learner,
	'top_number': args.top_number,
	'num_test_neg': num_test_neg
	}

	# NeuMF_PLUS
	NeuMF_PLUS_model = "NEUMF_PLUS_{:02d}node_{:02d}fac_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(layers[0], mf_dim, num_train_neg, num_test_neg, args.top_number, args.dataset, str(time()))
	# Create the Estimator
	neumf_plus_model = tf.estimator.Estimator(
	  model_fn=get_neuMF_PLUS_model, model_dir=model_save+"Models/new/NeuMF_PLUS/"+NeuMF_PLUS_model, params=params)

	# MLP_PLUS
	MLP_PLUS_model = "MLP_PLUS_{:02d}node_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(layers[0], num_train_neg, num_test_neg, args.top_number, args.dataset, str(time()))
	# Create the Estimator
	mlp_plus_model = tf.estimator.Estimator(
	  model_fn=get_MLP_PLUS_model, model_dir=model_save+"Models/new/MLP_PLUS/"+MLP_PLUS_model, params=params)

	# MF model
	MF_model = "MF_{:02d}fac_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(mf_dim, num_train_neg, num_test_neg, args.top_number, args.dataset, str(time()))
	# Create the Estimator
	mf_model = tf.estimator.Estimator(
	  model_fn=get_MF_model, model_dir=model_save+"Models/new/GMF/"+MF_model, params=params)

	# MLP model
	MLP_model = "MLP_{:02d}node_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(layers[0], num_train_neg, num_test_neg, args.top_number, args.dataset,str(time()))
	# Create the Estimator
	mlp_model = tf.estimator.Estimator(
	  model_fn=get_MLP_model, model_dir=model_save+"Models/new/MLP/"+MLP_model, params=params)

	# NeuMF model
	NeuMF_model = "NEUMF_{:02d}node_{:02d}fac_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(layers[0], mf_dim, num_train_neg, num_test_neg, args.top_number, args.dataset, str(time()))
	# Create the Estimator
	neumf_model = tf.estimator.Estimator(
	  model_fn=get_neuMF_model, model_dir=model_save+"Models/new/NeuMF/"+NeuMF_model, params=params)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "logits_relu"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)

	feature_eval, user_eval, item_eval, labels_eval = sample_plus.get_test_negative_instances_ver2(train,testRatings, feature_arr, num_test_neg, seed)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={
		"user_input": user_eval,
		"item_input": item_eval,
		"feature_input": feature_eval
		},
		y=labels_eval,
		batch_size=num_test_neg+4,
		num_epochs=1,
		shuffle=False)

	for i in range(num_epochs):
		
		feature_input, user_input, item_input, labels = sample_plus.get_train_instances(train, feature_arr, num_train_neg)
		t1 = time()
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={
			"user_input": user_input,
			"item_input": item_input,
			"feature_input": feature_input
			},
			y=labels,
			batch_size=batch_size,
			num_epochs=1,
			shuffle=True)

		neumf_plus_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t2 = time()
		print("Finished training NeuMF_PLUS model epoch {} in {:.2f} seconds".format(i,t2-t1))
		NeuMF_eval_results = neumf_plus_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t2))
		print(NeuMF_eval_results)


		mlp_plus_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t3= time()
		print("Finished training MLP_PLUS model epoch {} in {:.2f} seconds".format(i,t3-t2))
		MLP_eval_results = mlp_plus_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t3))
		print(MLP_eval_results)

		mf_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t4= time()
		print("Finished training MF model epoch {} in {:.2f} seconds".format(i,t4-t3))
		MF_eval_results = mf_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t4))
		print(MF_eval_results)

		mlp_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t5= time()
		print("Finished training MLP model epoch {} in {:.2f} seconds".format(i,t5-t4))
		MLP_eval_results = mlp_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t5))
		print(MLP_eval_results)

		neumf_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t6= time()
		print("Finished training NeuMF model epoch {} in {:.2f} seconds".format(i,t6-t5))
		NeuMF_eval_results = neumf_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t6))
		print(NeuMF_eval_results)


if __name__ == "__main__":
	tf.app.run()