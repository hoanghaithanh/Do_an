2013#Refactor code

import tensorflow as tf
import numpy as np
from Dataset import Dataset
import argparse
from time import time
def parse_args():
	parser = argparse.ArgumentParser(description="Run NeuMF.")
	parser.add_argument('--path', nargs='?', default='../../Data/ml-20m/',
						help='Input data path.')
	parser.add_argument('--dataset', nargs='?', default='ml-2m',
						help='Choose a dataset.')
	parser.add_argument('--epochs', type=int, default=100,
						help='Number of epochs.')
	parser.add_argument('--batch_size', type=int, default=256,
						help='Batch size.')
	parser.add_argument('--num_factors', type=int, default=9,
						help='Embedding size of MF model.')			
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


def get_MF_model(features, labels, mode, params):

	num_users = params['num_users']
	num_items = params['num_items']
	mf_dim = params['mf_dim']
	learning_rate = params['learning_rate']
	learner = params['learner']
	top_Number = params['top_number']
	num_test_neg = params['num_test_neg']

	#Input layers
	user_input = features['user_input']
	item_input = features['item_input']

	#############              MF parts              #############

	#Latent layers
	MF_user_input_variable = tf.Variable(tf.random_normal([num_users, mf_dim], stddev = 0.01))
	MF_user_embeded = tf.nn.embedding_lookup(MF_user_input_variable, user_input)

	MF_item_input_variable = tf.Variable(tf.random_normal([num_items, mf_dim], stddev = 0.01))
	MF_item_embeded = tf.nn.embedding_lookup(MF_item_input_variable, item_input)

	####MF prediction
	MF_predict_vector = tf.multiply(tf.layers.flatten(MF_user_embeded), tf.layers.flatten(MF_item_embeded))


	logits = tf.layers.Dense(units=1, name="prediction", activation=tf.sigmoid)(MF_predict_vector)

	predictions = {
	# Generate predictions (for PREDICT and EVAL mode)
	#"classes": tf.argmax(input=logits, axis=1),
	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	# `logging_hook`.
	#"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	"probabilities": tf.nn.relu(logits, name="logits_relu")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.log_loss(labels,logits)

	# configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		if learner.lower() == "adagrad": 
			optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
		elif learner.lower() == "rmsprop":
			optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
		elif learner.lower() == "adam":
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		else:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	tranformed_label = tf.reshape(tf.transpose(tf.where(tf.equal(labels, 1))),(1,-1))
	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	"recall": tf.metrics.recall_at_k(
		labels=tranformed_label, predictions=tf.transpose(logits),k=top_Number),
	"precision": tf.metrics.precision_at_k(
	 	labels=tranformed_label, predictions=tf.transpose(logits),k=top_Number)
	}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_label(rating):
	label = np.zeros(9)
	label[int((rating-1)*2)-1] = 1
	return label

def get_train_instances(train, num_train_neg):
	user_input, item_input, labels = [],[],[]
	num_users = train.shape[0]
	num_items = train.shape[1]
	for (u,i) in train.keys():
		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
		for k in range(num_train_neg):
			j = np.random.randint(num_items)
			while (u,j) in train.keys():
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')

	return user_arr, item_arr, labels_arr

def get_test_negative_instances(train, test, num_test_neg, seed):
	np.random.seed(seed)
	user_input, item_input, labels = [],[],[]
	num_items = train.shape[1]
	for entry in test:
		u = entry[0]
		i = entry[1]
		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
		for k in range(num_test_neg):
			j = np.random.randint(num_items)
			while (u,j) in train.keys():
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')
	return user_arr, item_arr, labels_arr

def get_test_negative_instances_ver2(train, test, num_test_neg, seed):
	np.random.seed(seed)
	user_input, item_input, labels = [],[],[]
	num_items = train.shape[1]
	current_user = -1
	for entry in test:
		u = entry[0]
		i = entry[1]
		if u != current_user:
			for k in range(num_test_neg):
				j = np.random.randint(num_items)
				while (u,j) in train.keys():
					j = np.random.randint(num_items)
				user_input.append(u)
				item_input.append(j)
				labels.append(0)
			current_user += 1

		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')
	return user_arr, item_arr, labels_arr

def main(unused_argv):
	args = parse_args()
	num_epochs = args.epochs
	batch_size = args.batch_size
	mf_dim = args.num_factors
	learning_rate = args.lr
	learner = args.learner
	num_train_neg = args.num_train_neg
	num_test_neg = args.num_test_neg
	
	dataset = Dataset(args.path + args.dataset)
	train, testRatings = dataset.trainMatrix, dataset.testRatings
	num_users, num_items = train.shape
	seed = args.seed

	params = {
	'num_users' : num_users,
	'num_items' : num_items,
	'mf_dim' : mf_dim,
	'learning_rate' : learning_rate,
	'learner' : learner,
	'top_number': args.top_number,
	'num_test_neg': num_test_neg
	}
	model = "MF{:02d}fac_{:02d}neg_{}topK_{}".format(mf_dim, num_train_neg, num_test_neg, str(time()))
	# Create the Estimator
	imp_MF_model = tf.estimator.Estimator(
	  model_fn=get_MF_model, model_dir="/Models/new/imp_MF_upgraded_model/"+model, params=params)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "logits_relu"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)

	# Evaluate the model and print results
	loss, recall, precision = [], [], []

	user_eval, item_eval, labels_eval = get_test_negative_instances_ver2(train,testRatings, num_test_neg, seed)
	print(item_eval)
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={
		"user_input": user_eval,
		"item_input": item_eval
		},
		y=labels_eval,
		batch_size=num_test_neg+4,
		num_epochs=1,
		shuffle=False)

	for i in range(num_epochs):
		t1 = time()
		user_input, item_input, labels = get_train_instances(train, num_train_neg)
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={
			"user_input": user_input,
			"item_input": item_input
			},
			y=labels,
			batch_size=batch_size,
			num_epochs=1,
			shuffle=True)

		imp_MF_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t2 = time()
		print("Finished training model epoch {} in {:.2f} second".format(i,t2-t1))
		eval_results = imp_MF_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t2))
		print(eval_results)

if __name__ == "__main__":
	tf.app.run()