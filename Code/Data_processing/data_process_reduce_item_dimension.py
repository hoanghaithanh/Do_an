# This code to reduce the item dimension, by generating mappting file
import numpy as np

item_array = np.full((131262),-2)
item_no = 0
fout = open("item_mapping","w")

with open("ratings_simplified.csv") as fin:
	for line in fin:
		tokens = line.split()
		item_array[int(tokens[1])-1] = -1
	for i in range(len(item_array)):
		if item_array[i]==-1:
			item_array[i]=item_no
			item_no+=1

for i in range(len(item_array)):
	if item_array[i]!=-2:
		fout.write("{0} {1} \n".format(i+1,item_array[i]))

fout.close()