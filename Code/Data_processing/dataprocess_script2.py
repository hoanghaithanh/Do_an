# This file is to simplify movies.csv file

# We do not care about movie title, year of release.etc.
# So that we eleminate them.
# We also code each genre as the number in Readme.txt file
# Each line in movies.csv contain atleast 1 integer
# The first integer is movie_id, the rest is code of genres
# this movie classified to.

def classify(word):
	if('Action' in word):
		return 1;
	if('Adventure' in word):
		return 2;
	if('Animation' in word):
		return 3;
	if("Children" in word):
		return 4;
	if('Comedy' in word):
		return 5;
	if('Crime' in word):
		return 6;
	if('Documentary' in word):
		return 7;
	if('Drama' in word):
		return 8;
	if('Fantasy' in word):
		return 9;
	if('Film-Noir' in word):
		return 10;
	if('Horror' in word):
		return 11;
	if('Musical' in word):
		return 12;
	if('Mystery' in word):
		return 13;
	if('Romance' in word):
		return 14;
	if('Sci-Fi' in word):
		return 15;
	if('Thriller' in word):
		return 16;
	if('War' in word):
		return 17;
	if('Western' in word):
		return 18;
	if('IMAX' in word):
		return 0;
fout = open("movie_simplified.csv", "w", encoding="utf8")
with open('movies.csv','r',encoding="utf8") as fin:
	fin.readline()
	for line in fin:
		# print(line)
		words = line.split(',')
		out_line = words[0]
		if(len(words)>=3):
			genres = words[len(words)-1].split('|')
			for genre in genres:
				out_line = out_line + ' ' + str(classify(genre))
		out_line += '\n'
		fout.write(out_line)
fout.close()
fin.close()



