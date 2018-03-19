fout = open("new_full-ml-20m.user.ident","w")
with open("full-ml-20m.user.ident") as fin:
	for line in fin:
		tokens = line.split()
		tokens[0] = str(int(tokens[0]) -1)
		out_line = ''
		for i in range(len(tokens)-1):
			out_line+=tokens[i]+' '
		out_line+=tokens[len(tokens)-1]+'\n'
		fout.write(out_line)
fout.close()