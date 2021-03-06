import numpy as np

def get_train_instances(train, num_train_neg):
	user_input, item_input, labels = [],[],[]
	num_users = train.shape[0]
	num_items = train.shape[1]
	for (u,i) in train.keys():
		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
		for k in range(num_train_neg):
			j = np.random.randint(num_items)
			while (u,j) in train.keys():
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')

	return user_arr, item_arr, labels_arr

def get_test_negative_instances(train, test, num_test_neg, seed):
	np.random.seed(seed)
	user_input, item_input, labels = [],[],[]
	num_items = train.shape[1]
	for entry in test:
		u = entry[0]
		i = entry[1]
		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
		for k in range(num_test_neg):
			j = np.random.randint(num_items)
			while (u,j) in train.keys():
				j = np.random.randint(num_items)
			user_input.append(u)
			item_input.append(j)
			labels.append(0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')
	return user_arr, item_arr, labels_arr

def get_test_negative_instances_ver2(train, test, num_test_neg, seed):
	np.random.seed(seed)
	user_input, item_input, labels = [],[],[]
	num_items = train.shape[1]
	current_user = -1
	for entry in test:
		u = entry[0]
		i = entry[1]
		if u != current_user:
			for k in range(num_test_neg):
				j = np.random.randint(num_items)
				while (u,j) in train.keys():
					j = np.random.randint(num_items)
				user_input.append(u)
				item_input.append(j)
				labels.append(0)
			current_user += 1

		user_input.append(u)
		item_input.append(i)
		labels.append(1.0)
	user_arr = np.array(user_input).reshape(-1,1)
	item_arr = np.array(item_input).reshape(-1,1)
	labels_arr = np.array(labels).reshape(-1,1).astype('float32')
	return user_arr, item_arr, labels_arr
