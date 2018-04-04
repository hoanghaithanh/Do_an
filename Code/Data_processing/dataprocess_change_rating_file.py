import numpy as np

item_array = np.full((131262),-2)

fout = open("../../Data/exp_ml-20m-full/modified_exp_ml-20m.train.rating","w")
with open("../../Data/item_mapping") as fin:
	for line in fin:
		tokens = line.split()
		item_array[int(tokens[0])-1] = int(tokens[1])
fin.close()

with open("../../Data/exp_ml-20m-full/exp_ml-20m.train.rating") as fin:
	for line in fin:
		tokens = line.split()
		if item_array[int(tokens[1])-1]==-2:
			print("something wrong!")
			break
		fout.write("\n{0} {1} {2} {3}".format(int(tokens[0]), item_array[int(tokens[1])-1], tokens[2], tokens[3]))
fout.close()