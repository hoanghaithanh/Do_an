#This file is to simplify rating.csv file

#for movielens is an explicit database while we
#only work with implicit data, we have to "downgrade"
#it, also simplify it
#Each line in rating_simplified.csv contains 2 int
#One for the user_id and one for the movie_id

fout = open("ratings_simplified.csv", "w")
fout2 = open("ml-20m.test.rating",'w')
with open('ratings.csv','r') as fin:
	fin.readline()
	lastitem = ['0','0','0','0']
	for line in fin:
		format='{0} {1}\n'
		words = line.split(',')
		if(words[0]!=lastitem[0]):
			out_line = format.format(lastitem[0],lastitem[1])
			fout2.write(out_line)
			lastitem=words
		else:
			if(int(lastitem[3])<int(words[3])):
				out_line = format.format(lastitem[0],lastitem[1])
				lastitem = words
			else:
				out_line = format.format(words[0],words[1])
			fout.write(out_line)
fout.close()
fin.close()


