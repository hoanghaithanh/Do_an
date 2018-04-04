#This file is to generate exp_ml-20m.train.rating and test file

fout = open("../../Data/exp_ml-20m-full/exp_ml-20m.train.rating", "w")
fout2 = open("../../Data/exp_ml-20m-full/exp_ml-20m.test.rating",'w')
with open('../../Data/ml-20m_Original/ratings.csv','r') as fin:
	
	lastitem = ['0','0','0','0']
	o_line = '{0} {1} {2} {3}'
	for i_line in fin:
		words = i_line.split(',')
		if(words[0]!=lastitem[0]):
			out_line = o_line.format(int(lastitem[0]), lastitem[1], lastitem[2], lastitem[3])
			fout2.write(out_line)
			lastitem=words
		else:
			if(int(lastitem[3])<int(words[3])):
				out_line = o_line.format(int(lastitem[0]), lastitem[1], lastitem[2], lastitem[3])
				lastitem = words
			else:
				out_line = o_line.format(int(words[0]), words[1], words[2], words[3])
	fout2.write(o_line.format(int(lastitem[0]), lastitem[1], lastitem[2], lastitem[3]))
fout.close()
fin.close()


