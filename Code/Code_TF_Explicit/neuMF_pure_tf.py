import tensorflow as tf
import numpy as np
from Dataset import Dataset
import argparse
import heapq
import math
import sys
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
	parser.add_argument('--layers', nargs='?', default='[72,36,18,9]',
						help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
	parser.add_argument('--reg_mf', type=float, default=0,
						help='Regularization for MF embeddings.')					
	parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
						help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
	parser.add_argument('--num_neg', type=int, default=4,
						help='Number of negative instances to pair with a positive instance.')
	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate.')
	parser.add_argument('--learner', nargs='?', default='adam',
						help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
	parser.add_argument('--verbose', type=int, default=1,
						help='Show performance per X iterations')
	parser.add_argument('--top_Number', type=int, default=10,
						help='Top K number')
	return parser.parse_args()

def getHitRatio(ranklist, gtItem):
	for item in ranklist:
		if item == gtItem:
			return 1
	return 0

def getNDCG(ranklist, gtItem):
	for i in range(len(ranklist)):
		item = ranklist[i]
		if item == gtItem:
			return math.log(2) / math.log(i+2)
	return 0

def get_train_instances(train, num_negatives, feature_arr):
	feature_input, user_input, item_input, labels = [],[],[],[]
	num_users = train.shape[0]
	for (u, i) in train.keys():
		# positive instance
		user_input.append(u)
		feature_input.append(feature_arr[u])
		item_input.append(i)
		labels.append(1)
		# negative instances
		for t in range(num_negatives):
			j = np.random.randint(num_items)
			while ((u, j) in train):
				j = np.random.randint(num_items)
			user_input.append(u)
			feature_input.append(feature_arr[u])
			item_input.append(j)
			labels.append(0)
	return feature_input, user_input, item_input, labels

args = parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
mf_dim = args.num_factors
layers = eval(args.layers)
reg_mf = args.reg_mf
reg_layers = eval(args.reg_layers)
num_negatives = args.num_neg
learning_rate = args.lr
learner = args.learner
verbose = args.verbose
k = args.top_Number
num_layer=len(layers)
t1 = time()

dataset = Dataset(args.path + args.dataset)
feature_arr, train, testRatings, testNegatives = dataset.feature_arr, dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape
fout = open("../../Result_log/NEUMF_pure_tf_Enhanced_{:02d}node_{:02d}fac_{:02d}neg_{}topK_{}".format(layers[0], mf_dim, num_negatives, k, str(time())),"w")
line = "NEUMF arguments: {} ".format(args)
print(line)
fout.write(line+'\n')
line=("Load data done [{:.1f} s]. #user={}, #item={}, #train={}, #test={}" 
	  .format(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
print(line)
fout.write(line+'\n')


feature_input = tf.placeholder(tf.float32, shape=(None,19),name="feature_input_layer")
user_input = tf.placeholder(tf.int32, shape=(None,1), name="user_input_layer")
item_input = tf.placeholder(tf.int32, shape=(None,1), name="item_input_layer")


MF_user_input_variable = tf.Variable(tf.random_normal([num_users, mf_dim], stddev = 0.01))
MF_user_embeded = tf.nn.embedding_lookup(MF_user_input_variable, user_input)
MF_item_input_variable = tf.Variable(tf.random_normal([num_items, mf_dim], stddev = 0.01))
MF_item_embeded = tf.nn.embedding_lookup(MF_item_input_variable, item_input)


MF_predict_vector = tf.multiply(tf.layers.flatten(MF_user_embeded), tf.layers.flatten(MF_item_embeded))

#MLP part
MLP_feature_dense = tf.layers.Dense(units=layers[0]/3, name="user_dense")(feature_input)

MLP_user_input_variable = tf.Variable(tf.random_normal([num_users, int(layers[0]/3)], stddev = 0.01))
MLP_user_embeded = tf.nn.embedding_lookup(MLP_user_input_variable, user_input)
MLP_item_input_variable = tf.Variable(tf.random_normal([num_items, int(layers[0]/3)], stddev = 0.01))
MLP_item_embeded = tf.nn.embedding_lookup(MLP_item_input_variable, item_input)

MLP_predict_vector = tf.concat([tf.layers.flatten(MLP_user_embeded), tf.layers.flatten(MLP_item_embeded), tf.layers.flatten(MLP_feature_dense)], 1)

for idx in range(1, num_layer):
	MLP_layer = tf.layers.Dense(layers[idx], kernel_regularizer= tf.keras.regularizers.l2(reg_layers[idx]), activation=tf.nn.relu, name = 'layer%d' %idx)
	MLP_predict_vector = MLP_layer(MLP_predict_vector)

NEU_predict_vector = tf.concat([MF_predict_vector,MLP_predict_vector], 1)

prediction = tf.layers.Dense(units=1, name="prediction", activation=tf.sigmoid)(NEU_predict_vector)

label = tf.placeholder(tf.float32, shape=(None,1), name="label")
loss = tf.keras.backend.binary_crossentropy(label,prediction,from_logits=False)

if learner.lower() == "adagrad": 
	train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
elif learner.lower() == "rmsprop":
	train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
elif learner.lower() == "adam":
	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
else:
	train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(num_epochs):
	feature_input_set, user_input_set, intem_input_set, label_set = get_train_instances(train, args.num_neg, feature_arr)
	user_input_array, item_input_arr, label_input_arr = np.array(user_input_set).reshape(-1,1), np.array(intem_input_set).reshape(-1,1), np.array(label_set).reshape(-1,1)
	t1=time()
	for batch in range(len(user_input_array)//args.batch_size):
		user_input_batch = user_input_array[batch*batch_size:min((batch+1)*batch_size,len(user_input_array))]
		feature_batch = feature_input_set[batch*batch_size:min((batch+1)*batch_size,len(feature_input_set))]
		item_batch = item_input_arr[batch*batch_size:min((batch+1)*batch_size,len(item_input_arr))]
		label_batch = label_input_arr[batch*batch_size:min((batch+1)*batch_size,len(label_input_arr))]

		sess.run(train_step, feed_dict={user_input: user_input_batch, feature_input: np.array(feature_batch), item_input: item_batch, label: label_batch})
	t2 = time()
	hits, ndcgs = [],[]
	for idx in range(len(testRatings)):
		rating = testRatings[idx]
		items = list(testNegatives[idx])
		u = rating[0]
		gtItem = rating[1]
		items.append(gtItem)
		# Get prediction scores
		map_item_score = {}
		features = np.full((len(items),19), feature_arr[u], dtype = 'float32')
		user_id = np.full((len(items)),rating[0],dtype='int32')
		predict = sess.run(prediction,feed_dict={user_input: np.array(user_id).reshape(-1,1), feature_input: features, item_input: np.array(items).reshape(-1,1)})
		for i in range(len(items)):
			item = items[i]
			map_item_score[item] = predict[i]
		items.pop()

		# Evaluate top rank list
		ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
		hr = getHitRatio(ranklist, gtItem)
		ndcg = getNDCG(ranklist, gtItem)
		hits.append(hr)
		ndcgs.append(ndcg)
	hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
	
	line = "Iteration {:d} [{:1f} s]: HR = {:4f}, NDCG = {:4f},  [{:1f} s]".format(epoch,  t2-t1, hr, ndcg, time()-t2)
	print(line)
	fout.write(line+'\n')
fout.close()