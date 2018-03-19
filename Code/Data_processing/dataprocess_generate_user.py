# This file is to generate user.csv file

# Each user is represented by a vector of 19 field which are stand for
# 19 genres.

import numpy as np
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="User description generation!")
	parser.add_argument('--numofuser', nargs='?', default=138493,
                        help='No. of user.')
	parser.add_argument('--numofmovie', nargs='?', default=131262,
                        help='No. of movie.')
	return parser.parse_args()

args = parse_args()
numofuser = args.numofuser
numofmovie = args.numofmovie
user_arr = np.zeros(shape=(numofuser,19),dtype=np.float64)
movie_arr = np.zeros(shape=(numofmovie,19),dtype=np.int32)

fmovie = open("movie_simplified.csv","r")
for line in fmovie:
	tokens = line.split(' ')
	if(tokens[1]=='None\n'):
		continue
	for k in tokens[1:]:
		movie_arr[int(tokens[0])-1, int(k)]=1
fmovie.close()
print("Finished load movie file!")

frating = open("ratings_simplified.csv","r")
for line in frating:
	tokens = line.split(' ')
	for i in range(19):
		if(movie_arr[int(tokens[1])-1, i]==1):
			user_arr[int(tokens[0])-1, i]+=1
frating.close()

fout = open("user_simplified.csv","w")
for i in range(numofuser):
	out_line = str(i+1)
	for k in user_arr[i]:
		out_line += ' ' + str(k)
	out_line+='\n'
	fout.write(out_line)
fout.close()

