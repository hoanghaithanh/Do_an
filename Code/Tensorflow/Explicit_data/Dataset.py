'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.num_users, self.num_items = self.trainMatrix.shape
        self.feature_arr = self.load_user_input_as_array(path+".user.ident")

    def load_user_input_as_array(self, filename):
        user_arr = np.zeros(shape=(self.num_users,19),dtype=np.float32)
        fin = open(filename, "r")
        for line in fin:
            if line != None and line != "":
                tokens = line.split()
                for k in range(1,20):
                    user_arr[int(tokens[0]),k-1] = float(tokens[k])
        return user_arr


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item, float(arr[2])])
                line = f.readline()
        return ratingList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split()
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = float(arr[2])
                line = f.readline()    
        return mat
