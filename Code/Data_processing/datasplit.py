import os
with open("../../Data/exp_ml-20m-full/exp_ml-20m.user.ident") as fin:
	file_count = 1
	user_count = 0
	o_line = '{} '*19+'{}\n'
	filepath = "../../Data/Data/exp_ml-20m_{}/exp_ml-20m_{}.user.ident"
	filename = filepath.format(1,1)
	filelist = []
	filelist.append(open(filename,"w"))
	for line in fin:
		token = line.split()
		# print(len(token))
		user_count=int(token[0])
		if user_count%10000 == 0 and (file_count - (user_count/10000)) == 0 and user_count > 0:
			filelist[-1].close()
			file_count+=1
			filename=filepath.format(file_count, file_count)
			filelist.append(open(filename,"w"))
		token[0] = int(int(token[0])%10000)
		outline = o_line.format(*token)
		filelist[-1].write(outline)
	filelist[-1].close()