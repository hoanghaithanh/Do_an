

fout = open("../../Data/exp_ml-20m-full/exp_ml-20m.train.rating", "w")
fout2 = open("../../Data/exp_ml-20m-full/exp_ml-20m.test.rating",'w')

train_mat = [[]]
with open('../../Data/ml-20m_Original/sorted_ratings.csv') as fin:
	count_user = 0
	for line in fin:
		tokens = line.split()
		if count_user < int(tokens[0]):
			train_mat.append([])
			count_user+=1
		train_mat[int(tokens[0])].append([int(tokens[1]), float(tokens[2]), int(tokens[3])])

	for i in range(count_user+1):
		for j in range(-1, -len(train_mat[i]), -1):
			if j>=-4:
				fout2.write("\n{0} {1} {2} {3}".format(i, train_mat[i][j][0], train_mat[i][j][1], train_mat[i][j][2]))
			else:
				fout.write("\n{0} {1} {2} {3}".format(i, train_mat[i][j][0], train_mat[i][j][1], train_mat[i][j][2]))

fout.close()
fout2.close()