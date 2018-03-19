import tensorflow as tf
import numpy as np
from Dataset import Dataset
import argparse
import heapq
import math
import sys
from time import time
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
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

def get_train_instances(train, num_negatives, user_arr):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(user_arr[u])
        item_input.append(i+1)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while ((u, j) in train):
                j = np.random.randint(num_items)
            user_input.append(user_arr[u])
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

args = parse_args()
path = args.path
dataset = args.dataset
layers = eval(args.layers)
reg_layers = eval(args.reg_layers)
num_negatives = args.num_neg
learner = args.learner
learning_rate = args.lr
batch_size = args.batch_size
epochs = args.epochs
verbose = args.verbose
num_layer = len(layers)
t1 = time()
dataset = Dataset(args.path + args.dataset)
user_arr, train, testRatings, testNegatives = dataset.user_arr, dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape
fout = open(str(time()),"w")
line = "MLP arguments: {} ".format(args)
print(line)
fout.write(line+'\n')
line=("Load data done [{:.1f} s]. #user={}, #item={}, #train={}, #test={}" 
      .format(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
print(line)
fout.write(line+'\n')


user_input = tf.placeholder(tf.float32, shape=(None,19),name="user_input_layer")
item_input = tf.placeholder(tf.int32, shape=(None,1), name="item_input_layer")

user_dense = tf.layers.Dense(units=layers[0]/2, name="user_dense")(user_input)

item_input_variable = tf.Variable(tf.random_normal([num_items, int(layers[0]/2)], stddev = 0.001))
item_embeded = tf.nn.embedding_lookup(item_input_variable, item_input)

vector = tf.concat([tf.layers.flatten(user_dense), tf.layers.flatten(item_embeded)],1)

for idx in range(1, num_layer):
    layer = tf.layers.Dense(layers[idx], kernel_regularizer= tf.keras.regularizers.l2(reg_layers[idx]), activation=tf.nn.relu, name = 'layer%d' %idx)
    vector = layer(vector)

prediction = tf.layers.Dense(units=1, name="prediction", activation=tf.sigmoid)(vector)

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
for epoch in range(epochs):
	user_in, item_set, label_set = get_train_instances(train, args.num_neg, user_arr)
	item_in, label_in = np.array(item_set).reshape(-1,1), np.array(label_set).reshape(-1,1)
	t1=time()
	for batch in range(len(user_in)//args.batch_size):
		user_batch = user_in[batch*batch_size:min((batch+1)*batch_size,len(user_in)-1)]
		item_batch = item_in[batch*batch_size:min((batch+1)*batch_size,len(item_in)-1)]
		label_batch = label_in[batch*batch_size:min((batch+1)*batch_size,len(label_in)-1)]
		sess.run(train_step, feed_dict={user_input: np.array(user_batch), item_input: item_batch, label: label_batch})
	t2 = time()
	hits, ndcgs = [],[]
	for idx in range(len(testRatings)):
		rating = testRatings[idx]
		items = list(testNegatives[idx])
		users = user_arr
		u = users[rating[0]]
		gtItem = rating[1]
		items.append(gtItem)
		# Get prediction scores
		map_item_score = {}
		users = np.full((len(items),19), u, dtype = 'int32')
		predict = sess.run(prediction,feed_dict={user_input: users, item_input: np.array(items).reshape(-1,1)})
		for i in range(len(items)):
		    item = items[i]
		    map_item_score[item] = predict[i]
		items.pop()

		# Evaluate top rank list
		ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
		hr = getHitRatio(ranklist, gtItem)
		ndcg = getNDCG(ranklist, gtItem)
		hits.append(hr)
		ndcgs.append(ndcg)
	hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
	
	line = "Iteration {:d} [{:.1f} s]: HR = {:.4f}, NDCG = {:.4f},  [{:.1f} s]".format(epoch,  t2-t1, hr, ndcg, time()-t2)
	print(line)
	fout.write(line+'\n')
fout.close()