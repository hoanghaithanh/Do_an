2013#Refactor code

import tensorflow as tf
import numpy as np
from Code_pure_TF_Added_FT.Dataset import Dataset
import argparse
from time import time
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF+.")
    parser.add_argument('--modelpath', nargs='?', default='../../Data/ml-20m/',
                        help='Model checkpoint path.')
    parser.add_argument('--path', nargs='?', default='../../Data/ml-20m/',
                        help='Model checkpoint path.')
    parser.add_argument('--dataset', nargs='?', default='ml-2m',
                        help='Choose a dataset.')
    parser.add_argument('--num_factors', type=int, default=9,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[72,36,18,9]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                   
    parser.add_argument('--reg_layers', nargs='?', default='[0.01,0.01,0.01,0.01]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--user', type=int, default=0,
                        help='User to recommend.')
    parser.add_argument('--top_number', type=int, default=10,
                        help='Top K number')
    return parser.parse_args()


def get_neuMF_plus_model(features, labels, mode, params):
    num_users = params['num_users']
    num_items = params['num_items']
    mf_dim = params['mf_dim']
    layers = params['layers']
    reg_layers = params['reg_layers']
    num_layer = len(layers)
    top_Number = params['top_number']

    #Input layers
    user_input = features['user_input']
    item_input = features['item_input']
    feature_input = features['feature_input']

    #############              MF parts              #############

    #Latent layers
    MF_user_input_variable = tf.Variable(tf.random_normal([num_users, mf_dim], stddev = 0.01))
    MF_user_embeded = tf.nn.embedding_lookup(MF_user_input_variable, user_input)

    MF_item_input_variable = tf.Variable(tf.random_normal([num_items, mf_dim], stddev = 0.01))
    MF_item_embeded = tf.nn.embedding_lookup(MF_item_input_variable, item_input)

    ####MF prediction
    MF_predict_vector = tf.multiply(tf.layers.flatten(MF_user_embeded), tf.layers.flatten(MF_item_embeded))


    #############              MLP parts             #############

    #MLP Laten layers
    MLP_user_input_variable = tf.Variable(tf.random_normal([num_users, int(layers[0]/3)], stddev = 0.01))
    MLP_user_embeded = tf.nn.embedding_lookup(MLP_user_input_variable, user_input)
    MLP_item_input_variable = tf.Variable(tf.random_normal([num_items, int(layers[0]/3)], stddev = 0.01))
    MLP_item_embeded = tf.nn.embedding_lookup(MLP_item_input_variable, item_input)

    #Dense feature
    MLP_feature_dense = tf.layers.Dense(units=layers[0]/3, name="user_dense", dtype=tf.float32)(feature_input)

    #MLP prediction
    MLP_predict_vector = tf.concat([tf.layers.flatten(MLP_user_embeded), tf.layers.flatten(MLP_item_embeded), tf.layers.flatten(MLP_feature_dense)], 1)

    #MLP layers
    for idx in range(1, num_layer):
        MLP_layer = tf.layers.Dense(layers[idx], kernel_regularizer= tf.contrib.layers.l2_regularizer(reg_layers[idx]), activation=tf.nn.relu, name='layer%d'%idx)
        MLP_predict_vector = MLP_layer(MLP_predict_vector)

    #NEU part
    NEU_predict_vector = tf.concat([MF_predict_vector,MLP_predict_vector], 1)

    logits = tf.layers.Dense(units=1, name="prediction", activation=tf.sigmoid)(NEU_predict_vector)

    predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    #"classes": tf.argmax(input=logits, axis=1),
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    "item": item_input,
    "probabilities": tf.nn.relu(logits, name="logits_relu")
    }


    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def recommend_K_item(model, train, user, features, K):
    item_list = []
    for i in range(train.shape[1]):
        if (user, i) not in train.keys():
            item_list.append(i)
    user_arr = np.full(len(item_list), user)
    item_arr = np.array(item_list)
    feature_arr = np.full((len(item_list),19), features[user])

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
        "user_input": user_arr,
        "item_input": item_arr,
        "feature_input": feature_arr
        },
        shuffle=False)
    predictions = model.predict(predict_input_fn)
    for p in predictions:
        print(p)


def main(unused_argv):
    args = parse_args()
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    dataset = Dataset(args.path + args.dataset)
    feature_arr, train = dataset.feature_arr, dataset.trainMatrix
    num_users, num_items = train.shape

    params = {
    'num_users' : num_users,
    'num_items' : num_items,
    'mf_dim' : mf_dim,
    'layers' : layers,
    'reg_layers' : reg_layers,
    'top_number': args.top_number
    }    # Create the Estimator
    imp_neuMF_model = tf.estimator.Estimator(
      model_fn=get_neuMF_plus_model, model_dir=args.modelpath, params=params)

    recommend_K_item(imp_neuMF_model, train, args.user, feature_arr, args.top_number)


if __name__ == "__main__":
    tf.app.run()