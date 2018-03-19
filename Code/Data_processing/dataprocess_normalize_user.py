# This file is to normalize user_simplified file in to user_normalized
# Each user is represented by a vector of 19 field which are stand for
# 19 genres, each field is a floating point number from 0 to 1.

fin = open("user_simplified.csv","r")
fout = open("user_normalized.csv","w")

for line in fin:
	tokens = line.split();
	max = float(tokens[1])
	for k in tokens[1:]:
		if(max<float(k)):
			max=float(k)
	for k in range(1,20):
		tokens[k] = "{:.4f}".format(float(tokens[k])/max)
	out_line=''
	for k in range(19):
		out_line+=str(tokens[k])+' '
	out_line+=str(tokens[19])+'\n'
	fout.write(out_line)
fin.close()
fout.close()