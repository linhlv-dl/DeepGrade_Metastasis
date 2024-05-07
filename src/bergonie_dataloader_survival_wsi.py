import os
import math
import torch
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold

class Patient(data_utils.Dataset):
	"""docstring for MNISTBags"""
	def __init__(self, root_folder, 
						indices_folder, 
						patient_list, 
						labels_list, 
						patients_wsi, 
						transf = None, 
						n_tiles = 10000, 
						seed = 2334, 
						region = None):
		super(Patient, self).__init__()
		self.root_folder = root_folder
		self.patient_list = patient_list
		self.labels_list = labels_list
		self.transforms = transf
		self.seed = seed
		self.region = region
		self.n_tiles = n_tiles
		self.indices_folder = indices_folder
		self.patients_wsi = patients_wsi
		self.list_of_bags = self.get_all_bags()
		

	def _read_folder(self, npz_file, label = 0):
		npz_array = np.load(npz_file)['arr_0']
		list_labels = [label] * npz_array.shape[0]
		return npz_array, np.array(list_labels), npz_array.shape[0]

	def random_select_tiles_wsi(self, np_array, num_tiles = 10000):
		if num_tiles == -1:
			return np_array

		n_rows = np_array.shape[0]
		rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
		selected_array = np_array[rd_indices,:]

		return selected_array
	
	def _setup_bag_2(self, pname, n_tiles = 10000): 
		#print("Get bag of {}".format(self.region))
		list_wsis = self.patients_wsi[pname]
		all_wsi = []
		for wsi in list_wsis:
			wsi_path = os.path.join(self.root_folder, wsi)
			c_files, _, _ = self._read_folder(wsi_path)
			
			# indices of region
			re_indices_path = os.path.join(self.indices_folder, wsi)
			re_indices = np.load(re_indices_path)['arr_0']
			print(wsi, re_indices.shape)
			
			c_files = c_files[re_indices,:]
			all_wsi.append(c_files)

		wsi_file = np.concatenate(all_wsi, axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)
		selected_tiles = self.random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
		print('Shape: ', wsi_file.shape, selected_tiles.shape)

		return selected_tiles

	def _setup_bag_region(self, pname, indices_region): 
		cur_indices_folder = os.path.join(self.indices_folder, indices_region, 'index')
		list_files = sorted(list(os.listdir(cur_indices_folder)))

		list_wsis = [f_name for f_name in list_files if pname in f_name]
		all_wsi = []
		for wsi in list_wsis:
			wsi_path = os.path.join(self.root_folder, wsi)
			c_files, _, _ = self._read_folder(wsi_path)
			
			# indices of region
			#indices_path = os.path.join(self.indices_foler, indices_region, 'index')
			re_indices_path = os.path.join(cur_indices_folder, wsi)
			re_indices = np.load(re_indices_path)['arr_0']
			print(wsi, re_indices.shape)
			
			c_files = c_files[re_indices,:]
			all_wsi.append(c_files)

		wsi_file = np.concatenate(all_wsi, axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)
		#selected_tiles = self.random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
		#print('Shape: ', wsi_file.shape, selected_tiles.shape)

		return wsi_file

	def _combination_2_regions(self, pname, region1, region2, n_tiles = 10000):
		c_files = self._setup_bag_region(pname, region1)
		c_files = self.random_select_tiles_wsi(c_files, num_tiles = int(n_tiles * 0.5))

		p_files = self._setup_bag_region(pname, region2)
		p_files = self.random_select_tiles_wsi(p_files, num_tiles = int(n_tiles * 0.5))

		wsi_file = np.concatenate((c_files, p_files), axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)

		return wsi_file

	def _combination_3_regions(self, pname, region1, region2, region3, n_tiles = 10000):
		c_files = self._setup_bag_region(pname, region1)
		c_files = self.random_select_tiles_wsi(c_files, num_tiles = int(n_tiles * 0.34))

		p_files = self._setup_bag_region(pname, region2)
		p_files = self.random_select_tiles_wsi(p_files, num_tiles = int(n_tiles * 0.33))

		k_files = self._setup_bag_region(pname, region3)
		k_files = self.random_select_tiles_wsi(k_files, num_tiles = int(n_tiles * 0.33))

		wsi_file = np.concatenate((c_files, p_files, k_files), axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)

		return wsi_file

	def _combination_4_regions(self, pname, n_tiles = 10000):
		c_files = self._setup_bag_region(pname, 'Center')
		c_files = self.random_select_tiles_wsi(c_files, num_tiles = int(n_tiles * 0.25))

		p_files = self._setup_bag_region(pname, 'Periphery')
		p_files = self.random_select_tiles_wsi(p_files, num_tiles = int(n_tiles * 0.25))

		k_files = self._setup_bag_region(pname, 'R1')
		k_files = self.random_select_tiles_wsi(k_files, num_tiles = int(n_tiles * 0.25))

		t_files = self._setup_bag_region(pname, 'TNormal')
		t_files = self.random_select_tiles_wsi(t_files, num_tiles = int(n_tiles * 0.25))

		wsi_file = np.concatenate((c_files, p_files, k_files, t_files), axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)

		return wsi_file

	def _setup_bag_combination(self, pname, region, n_tiles = 10000): 
		print("Get bag of {}".format(self.region))

		if region == 'CP':
			return self._combination_2_regions(pname, "Center", "Periphery", n_tiles = 10000)
		if region == 'CR1':
			return self._combination_2_regions(pname, "Center", "R1", n_tiles = 10000)
		if region == 'CTNormal':
			return self._combination_2_regions(pname, "Center", "TNormal", n_tiles = 10000)
		if region == 'PR1':
			return self._combination_2_regions(pname, "Periphery", "R1", n_tiles = 10000)
		if region == 'PTNormal':
			return self._combination_2_regions(pname, "Periphery", "TNormal", n_tiles = 10000)
		if region == 'R1TNormal':
			return self._combination_2_regions(pname, "R1", "TNormal", n_tiles = 10000)
		if region == 'CPR1':
			return self._combination_3_regions(pname, "Center", "Periphery", "R1", n_tiles = 10000)
		if region == 'CPTNormal':
			return self._combination_3_regions(pname, "Center", "Periphery", "TNormal", n_tiles = 10000)
		if region == 'CR1TNormal':
			return self._combination_3_regions(pname, "Center", "R1", "TNormal", n_tiles = 10000)
		if region == 'PR1TNormal':
			return self._combination_3_regions(pname, "Periphery", "R1", "TNormal", n_tiles = 10000)
		if region == 'All':
			return self._combination_4_regions(pname, n_tiles = 10000)
		return None
		

	def get_all_bags(self):
		all_bags = []
		for idx in range(len(self.patient_list)):
			patient = self.patient_list[idx]
			status, survtime = self.labels_list[idx]
			
			if self.region in ['Center', 'Periphery', 'R1', 'TNormal']:
				# Get data and load 
				bags = self._setup_bag_2(patient, self.n_tiles)
			else:
				self.patients_wsi = None
				bags = self._setup_bag_combination(patient, self.region, self.n_tiles)

			# Convert to tensor
			bags_tensor = torch.from_numpy(bags)
			status = torch.Tensor([status])
			survtime = torch.Tensor([survtime])
			sample = {'bags_tensor': bags_tensor, 
						'istatus': status,
						'isurvtime':survtime}

			if self.transforms is not None:
				sample = self.transforms(sample)
			all_bags.append((sample['bags_tensor'], 
								sample['istatus'], 
								sample['isurvtime']))
		return all_bags

	def __len__(self):
		return len(self.labels_list)

	def __getitem__(self, index):
		return self.list_of_bags[index]

def read_csv(csv_file, wsi_folder):
	# Read WSI folder
	list_files = sorted(list(os.listdir(wsi_folder)))
	# Read CSV file
	file = open(csv_file)
	csvreader = csv.reader(file)
	header = next(csvreader)
	patients = {}
	patients_wsi = {}
	for row in csvreader:
		name, status, year = row
		patients[name] = (int(status), float(year))
		wsi_files = [f_name for f_name in list_files if name in f_name]
		patients_wsi[name] = wsi_files
	file.close()
	return patients, patients_wsi

def search_keys(npz_name, labels_keys):
	for x_key in labels_keys:
		if x_key in npz_name:
			return x_key 
	return None

def weighted_sampler(target_labels):
	labels_count = np.unique(target_labels, return_counts = True)[1]
	class_weight = 1./ labels_count
	
	samples_weight = class_weight[target_labels]
	samples_weight = torch.from_numpy(samples_weight)
	samples_weight = samples_weight.double()
	sampler = data_utils.WeightedRandomSampler(samples_weight, len(samples_weight))
	return sampler

def get_data_npz_folder(root_folder, 
	csv_labels, 
	indices_folder,
	n_tiles = 0,
	batch_size = 1, 
	val_per = 0.2,
	seed = 2334,
	gregion = None):
	
	# Read labels file
	labels_dict, patients_wsi = read_csv(csv_labels, indices_folder)
	list_patients = list(labels_dict)
	
	num_samples = len(list_patients)
	indices = list(range(num_samples))
	split = max(int(np.floor(val_per * num_samples)), 1)
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]

	# Read sample files and split
	train_patients = [list_patients[t] for t in train_idx]
	train_labels = []
	for tdx in train_idx:
		pname = list_patients[tdx]
		train_labels.append(labels_dict[pname])

	valid_patients = [list_patients[v] for v in valid_idx]
	valid_labels = []
	for vdx in valid_idx:
		pname = list_patients[vdx]
		valid_labels.append(labels_dict[pname])

	# Test on 3 patients
	# train_patients = train_patients[:3]
	# train_labels = train_labels[:3]
	# valid_patients = valid_patients[:1]
	# valid_labels = valid_labels[:1]

	#train_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	train_transf = None
	train_dataset = Patient(root_folder, 
							indices_folder,
							train_patients, 
							train_labels, 
							patients_wsi,
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							region = gregion)
	train_status = list(list(zip(*train_labels))[0])
	train_weighted_sampler = weighted_sampler(train_status)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=0,
													sampler = train_weighted_sampler)

	#valid_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							indices_folder,
							valid_patients, 
							valid_labels, 
							patients_wsi,
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							region = gregion)
	#valid_status = list(list(zip(*valid_labels))[0])
	#valid_weighted_sampler = weighted_sampler(valid_status)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													num_workers = 0, 
													shuffle = False,)
													#sampler = valid_weighted_sampler)

	print("Total training patients....: {}".format(len(train_dataset)))
	print("Total validation patients....: {}".format(len(valid_dataset)))
	valid_patients_dict = {}
	valid_patients_dict['root'] = root_folder
	valid_patients_dict['patients'] = valid_patients
	valid_patients_dict['labels'] = valid_labels

	return train_dataloader, valid_dataloader, valid_patients_dict

def get_data_cross_validation_KFold(csv_labels, root_folder, region_indices, k_folds = 5, seed = 2334):
	# Read labels file
	labels_dict, patients_wsi = read_csv(csv_labels, region_indices)
	list_patients = list(labels_dict)

	#(int(status), float(year))
	patients_labels = []
	for pname in list_patients:
		#pname = list_patients[vdx]
		patients_labels.append(labels_dict[pname])

	patients =  np.array(list_patients)
	patients_labels =  np.array(patients_labels)
	
	kf = KFold(n_splits = k_folds, shuffle=True, random_state = seed)
	train_loaders = []
	valid_loaders = []
	train_patients_dict = {}
	valid_patients_dict = {}
	fold_idx = 0
	for train_index, valid_index in kf.split(patients):
		train_patients, valid_patients = patients[train_index], patients[valid_index]
		train_labels, valid_labels = patients_labels[train_index], patients_labels[valid_index]
		#print(np.unique(np.array(testing_patients), return_counts = True))

		train_patients_dict['fold_' + str(fold_idx)] = (train_patients, train_labels)
		valid_patients_dict['fold_' + str(fold_idx)] = (valid_patients, valid_labels)
		fold_idx += 1

	return train_patients_dict, valid_patients_dict, patients_wsi

def get_data_cross_validation_SKFold(csv_labels, root_folder, region_indices, k_folds = 5, seed = 2334):
	# Read labels file
	labels_dict, patients_wsi = read_csv(csv_labels, region_indices)
	list_patients = list(labels_dict)

	patients_labels = []
	for pname in list_patients:
		patients_labels.append(labels_dict[pname])

	patients =  np.array(list_patients)
	patients_labels =  np.array(patients_labels)
	events = patients_labels[:,0]
	events = np.array(events)
	
	kf = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state = seed)
	train_loaders = []
	valid_loaders = []
	train_patients_dict = {}
	valid_patients_dict = {}
	fold_idx = 0
	for train_index, valid_index in kf.split(patients, events):
		train_patients, valid_patients = patients[train_index], patients[valid_index]
		train_labels, valid_labels = patients_labels[train_index], patients_labels[valid_index]
		print('Fold {} = {}'.format(fold_idx,sum(events[valid_index])))

		train_patients_dict['fold_' + str(fold_idx)] = (train_patients, train_labels)
		valid_patients_dict['fold_' + str(fold_idx)] = (valid_patients, valid_labels)
		fold_idx += 1

	return train_patients_dict, valid_patients_dict, patients_wsi

def get_data_from_cross_validation(root_folder, 
	indices_folder,
	train_dict,
	valid_dict, 
	patients_wsi,
	n_tiles = 0,
	batch_size = 1, 
	seed = 2334,
	gregion = None):
	
	# Read sample files and split
	print("Load the loaders .........")
	train_patients, train_labels = train_dict[0], train_dict[1]
	valid_patients, valid_labels = valid_dict[0], valid_dict[1]

	#train_patients = train_patients[:5]
	#train_labels = train_labels[:5]
	#valid_patients = valid_patients[:3]
	#valid_labels = valid_labels[:3]

	train_transf = None
	train_dataset = Patient(root_folder, 
							indices_folder,
							train_patients, 
							train_labels,
							patients_wsi, 
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							region = gregion)
	train_status = list(list(zip(*train_labels))[0])
	train_status = list(map(int, train_status)) # convert list of float number to in
	train_weighted_sampler = weighted_sampler(train_status)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=20,
													sampler = train_weighted_sampler)

	#valid_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							indices_folder,
							valid_patients, 
							valid_labels, 
							patients_wsi,
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							region = gregion)
	valid_status = list(list(zip(*valid_labels))[0])
	valid_status = list(map(int, valid_status))
	#valid_weighted_sampler = weighted_sampler(valid_status)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													num_workers = 20, 
													shuffle = False,)
													#sampler = valid_weighted_sampler)

	print("Total training patients....: {}".format(len(train_dataset)))
	print("Total validation patients....: {}".format(len(valid_dataset)))

	return train_dataloader, valid_dataloader
