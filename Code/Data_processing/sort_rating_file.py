import numpy as np

fout = open('../../Data/ml-20m_Original/sorted_ratings.csv','w')

train_mat = [[]]
with open('../../Data/ml-20m_Original/ratings.csv','r') as fin:
	fin.readline()
	count_user = 0
	for line in fin:
		tokens = line.split(",")
		if count_user < int(tokens[0])-1:
			train_mat.append([])
			count_user+=1
		train_mat[int(tokens[0])-1].append([int(tokens[1]), float(tokens[2]), int(tokens[3])])

	for i in range(count_user+1):
		train_mat[i].sort(key=lambda student: student[2])
	
	for i in range(count_user+1):
		for j in range(len(train_mat[i])):
			fout.write("\n{0} {1} {2} {3}".format(i, train_mat[i][j][0], train_mat[i][j][1], train_mat[i][j][2]))
fout.close()