# This file is to generate testing file from training filr

import scipy.sparse as sp
import numpy as np
fin = open("new_ratings_simplified.csv")

mat = sp.dok_matrix((138492, 26744), dtype=np.float32)
fin.readline()
for line in fin:
	tokens = line.split()
	mat[int(tokens[0])-1,int(tokens[1])-1] = 1
fin.close()
fout = open("new_ml-20m.test.negative","w")
for user in range(138492):
	negative_items = []
	while(len(negative_items)<100):
		item = np.random.randint(0,26743)
		while(mat[user, item]==1):
			item = np.random.randint(0,26743)
		negative_items.append(item)
	out_line = str(user)
	for item in negative_items:
		out_line+=' '+str(item)
	out_line+='\n'
	fout.write(out_line)
fout.close()