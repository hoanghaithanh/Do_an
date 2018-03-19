import numpy as np

item_array = np.full((131262),-2)

fout = open("new_ratings_simplified.csv","w")
with open("item_mapping") as fin:
	for line in fin:
		tokens = line.split()
		item_array[int(tokens[0])-1] = int(tokens[1])
fin.close()

with open("ratings_simplified.csv") as fin:
	for line in fin:
		tokens = line.split()
		if item_array[int(tokens[1])-1]==-2:
			print("something wrong!")
			break
		fout.write("{0} {1} \n".format(int(tokens[0])-1, item_array[int(tokens[1])-1]))
fout.close()