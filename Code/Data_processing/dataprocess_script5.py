# This file is to normalize user_simplified file in to user_normalized
# Each user is represented by a vector of 19 field which are stand for
# 19 genres, each field is a floating point number from 0 to 1.

import scipy.sparse as sp
import numpy as np
fin = open("ratings.csv")

mat = sp.dok_matrix((138493, 131262), dtype=np.float32)
fin.readline()
for line in fin:
	tokens = line.split(',')
	mat[int(tokens[0])-1,int(tokens[1])-1] = 1
fin.close()
fout = open("ml-20m.test.negative","w")
for user in range(138493):
	negative_items = []
	while(len(negative_items)<100):
		item = np.random.randint(1,131262)
		while(mat[user, item]==1):
			item = np.random.randint(1,131262)
		negative_items.append(item)
	out_line = str(user+1)
	for item in negative_items:
		out_line+=' '+str(item)
	out_line+='\n'
	fout.write(out_line)
fout.close()