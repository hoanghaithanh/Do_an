if __name__ == '__main__':
	import tensorflow as tf
	import numpy as np
	from Dataset import Dataset
	import sample
	import argparse
	from time import time
else:
	import tensorflow as tf
	import numpy as np
	from .Dataset import Dataset
	from . import sample
	import argparse
	from time import time
def parse_args():
	parser = argparse.ArgumentParser(description="Run MLP.")
	parser.add_argument('--dir_path', nargs='?', default='/',
						help='Model save path')
	parser.add_argument('--path', nargs='?', default='../../Data/ml-20m/',
						help='Input data path.')
	parser.add_argument('--dataset', nargs='?', default='ml-2m',
						help='Choose a dataset.')
	parser.add_argument('--epochs', type=int, default=50,
						help='Number of epochs.')
	parser.add_argument('--batch_size', type=int, default=128,
						help='Batch size.')
	parser.add_argument('--layers', nargs='?', default='[256,128,64,32]',
						help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")			
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


def get_MLP_model(features, labels, mode, params):

	num_users = params['num_users']
	num_items = params['num_items']
	layers = params['layers']
	reg_layers = params['reg_layers']
	learning_rate = params['learning_rate']
	learner = params['learner']
	num_layer = len(layers)
	top_Number = params['top_number']
	num_test_neg = params['num_test_neg']

	#Input layers
	user_input = features['user_input']
	item_input = features['item_input']

	#############              MF parts              #############

	#Latent layers
		#MLP Laten layers
	MLP_user_input_variable = tf.Variable(tf.random_normal([num_users, int(layers[0]/2)], stddev = 0.01))
	MLP_user_embeded = tf.nn.embedding_lookup(MLP_user_input_variable, user_input)
	MLP_item_input_variable = tf.Variable(tf.random_normal([num_items, int(layers[0]/2)], stddev = 0.01))
	MLP_item_embeded = tf.nn.embedding_lookup(MLP_item_input_variable, item_input)

	#Dense feature
	# MLP_feature_dense = tf.layers.Dense(units=layers[0]/3, name="user_dense", dtype=tf.float32)(feature_input)

	#MLP prediction
	MLP_predict_vector = tf.concat([tf.layers.flatten(MLP_user_embeded), tf.layers.flatten(MLP_item_embeded)], 1)

	#MLP layers
	for idx in range(1, num_layer):
		MLP_layer = tf.layers.Dense(layers[idx], kernel_regularizer= tf.contrib.layers.l2_regularizer(reg_layers[idx]), activation=tf.nn.relu, name='layer%d'%idx)
		MLP_predict_vector = MLP_layer(MLP_predict_vector)


	logits = tf.layers.Dense(units=1, name="prediction", activation=tf.sigmoid)(MLP_predict_vector)

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
		labels=tranformed_label, predictions=tf.transpose(logits),k=top_Number)
	}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	args = parse_args()
	model_dir=args.dir_path
	num_epochs = args.epochs
	batch_size = args.batch_size
	layers = eval(args.layers)
	reg_layers = eval(args.reg_layers)
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
	'layers' : layers,
	'reg_layers' : reg_layers,
	'learning_rate' : learning_rate,
	'learner' : learner,
	'top_number': args.top_number,
	'num_test_neg': num_test_neg
	}
	model = "MLP_{:02d}node_{:02d}trainneg_{:02d}testneg_{:02d}topK_{}dataset_{}".format(layers[0], num_train_neg, num_test_neg, args.top_number, args.dataset,str(time()))
	# Create the Estimator
	imp_MLP_model = tf.estimator.Estimator(
	  model_fn=get_MLP_model, model_dir=model_dir+"Models/new/MLP/"+model, params=params)

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "logits_relu"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)

	# Evaluate the model and print results
	loss, recall, precision = [], [], []

	user_eval, item_eval, labels_eval = sample.get_test_negative_instances_ver2(train,testRatings, num_test_neg, seed)
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
		user_input, item_input, labels = sample.get_train_instances(train, num_train_neg)
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={
			"user_input": user_input,
			"item_input": item_input
			},
			y=labels,
			batch_size=batch_size,
			num_epochs=1,
			shuffle=True)

		imp_MLP_model.train(
			input_fn=train_input_fn,
		  	steps=40000,
			hooks=[logging_hook])
		t2 = time()
		print("Finished training model epoch {} in {:.2f} second".format(i,t2-t1))
		eval_results = imp_MLP_model.evaluate(input_fn=eval_input_fn)
		print("Finished testing model in {:.2f} second".format(time()-t2))

if __name__ == "__main__":
	tf.app.run()