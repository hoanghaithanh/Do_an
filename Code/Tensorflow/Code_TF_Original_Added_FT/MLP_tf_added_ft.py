'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import tensorflow as tf
import numpy as np
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[96,48,24,12]',
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

def init_normal(shape, name=None, dtype=None, partition_info=None):
    return tf.keras.initializers.RandomNormal(stddev=0.01)(shape, partition_info=partition_info)

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name = 'item_input')
    feature_input = tf.keras.layers.Input(shape=(19,), dtype='float32', name = 'feature')

    MLP_Embedding_User = tf.keras.layers.Embedding(input_dim = num_users, output_dim = layers[0]/3, name = 'user_embedding',
                                  embeddings_initializer = init_normal, embeddings_regularizer = tf.keras.regularizers.l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim = num_items, output_dim = layers[0]/3, name = 'item_embedding',
                                  embeddings_initializer = init_normal, embeddings_regularizer = tf.keras.regularizers.l2(reg_layers[0]), input_length=1)   
    MLP_Dense_Feature = tf.keras.layers.Dense(units=layers[0]/3, kernel_regularizer=tf.keras.regularizers.l2(reg_layers[0]), input_shape=(19,))
    # Crucial to flatten an embedding vector!
    user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
    item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
    feature = tf.keras.layers.Flatten()(MLP_Dense_Feature(feature_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = tf.keras.layers.Concatenate()([user_latent, item_latent, feature])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = tf.keras.layers.Dense(layers[idx], kernel_regularizer= tf.keras.regularizers.l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = tf.keras.models.Model(inputs=[feature_input, user_input, item_input], 
                  outputs=prediction)
    
    return model

def get_train_instances(train, num_negatives, feature_arr):
    feature_input, user_input, item_input, labels = [],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        feature_input.append(feature_arr[u])
        labels.append(1)
        # negative instancesd
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            feature_input.append(feature_arr[u])
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return feature_input, user_input, item_input, labels

if __name__ == '__main__':
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
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    feature_arr, train, testRatings, testNegatives = dataset.feature_arr, dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), loss='binary_crossentropy')    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, feature_arr, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        feature_input, user_input, item_input, labels = get_train_instances(train, num_negatives, feature_arr)
    
        # Training        
        hist = model.fit([np.array(feature_input), np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True, callbacks=[tensorboard])
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, feature_arr, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
