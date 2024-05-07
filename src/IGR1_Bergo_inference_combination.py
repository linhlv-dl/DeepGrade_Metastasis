import sys
import os
import torch.nn
import torch 
from torchvision import transforms

from model_attention_survival import NN_Model2a
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.utils import shuffle
from PIL import Image
from datetime import datetime
import pickle
import csv

def inference_validation(ckpt_path, root_folder, indices_folder, labels_dict, comb = 'CP'):
    patient_list = list(labels_dict)
    print(patient_list)
    wsis_list = get_wsis_per_patient(patient_list, indices_folder)

    checkpoint = torch.load(ckpt_path)
    for key in list(checkpoint['state_dict'].keys()):
    	new_key = key[6:]
    	checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)

    model = NN_Model2a(fc_1= 256, fc_2 = 128)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    predictions = {}
    for ptx in range(len(patient_list)):
        patient = patient_list[ptx]
        label_dict = labels_dict[patient]

        #wsi_file = _setup_bag(root_folder, indices_folder, wsis_list, patient, comb)
        wsi_file = _setup_bag_combination(root_folder, indices_folder, patient, comb, n_tiles = 10000)
        input_tensor = torch.from_numpy(wsi_file)
        input_tensor = input_tensor.unsqueeze(0)
        inf_patient = input_tensor
    	
    	# predict
        y_prob, x_att = model(inf_patient)
        predictions[patient] = y_prob.squeeze().item()
        #predictions.append(y_prob.squeeze().item())

    return predictions

def get_wsis_per_patient(patient_list, indices_folder):
    list_files = sorted(list(os.listdir(indices_folder)))
    patients_wsi = {}
    for patient in patient_list:
        wsi_files = [f_name for f_name in list_files if patient in f_name]
        patients_wsi[patient] = wsi_files
    return patients_wsi

def _read_folder(npz_file, label = 0):
    npz_array = np.load(npz_file)['arr_0']
    list_labels = [label] * npz_array.shape[0]

    return npz_array, np.array(list_labels), npz_array.shape[0]

def random_select_tiles(np_array, num_tiles = 10000):
    n_rows = np_array.shape[0]
    rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
    selected_array = np_array[rd_indices,:]

    return selected_array

def _setup_bag_region(root_folder, indices_folder, pname, indices_region, seed = 2452): 
    cur_indices_folder = os.path.join(indices_folder, indices_region, 'index')
    list_files = sorted(list(os.listdir(cur_indices_folder)))

    list_wsis = [f_name for f_name in list_files if pname in f_name]
    all_wsi = []
    for wsi in list_wsis:
        wsi_path = os.path.join(root_folder, wsi)
        c_files, _, _ = _read_folder(wsi_path)
            
        # indices of region
        #indices_path = os.path.join(self.indices_foler, indices_region, 'index')
        re_indices_path = os.path.join(cur_indices_folder, wsi)
        re_indices = np.load(re_indices_path)['arr_0']
        print(wsi, c_files.shape, re_indices.shape)
            
        c_files = c_files[re_indices,:]
        all_wsi.append(c_files)

    wsi_file = np.concatenate(all_wsi, axis = 0)
    wsi_file= shuffle(wsi_file, random_state = seed)

    return wsi_file

def _combination_2_regions(root_folder, indices_folder, pname, region1, region2, n_tiles = 10000, seed = 2452):
    c_files = _setup_bag_region(root_folder, indices_folder, pname, region1)
    c_files = random_select_tiles(c_files, num_tiles = int(n_tiles * 0.5))

    p_files = _setup_bag_region(root_folder, indices_folder, pname, region2)
    p_files = random_select_tiles(p_files, num_tiles = int(n_tiles * 0.5))

    wsi_file = np.concatenate((c_files, p_files), axis = 0)
    wsi_file= shuffle(wsi_file, random_state = seed)

    return wsi_file

def _combination_3_regions(root_folder, indices_folder, pname, region1, region2, region3, n_tiles = 10000, seed = 2452):
    c_files = _setup_bag_region(root_folder, indices_folder, pname, region1)
    c_files = random_select_tiles(c_files, num_tiles = int(n_tiles * 0.34))

    p_files = _setup_bag_region(root_folder, indices_folder, pname, region2)
    p_files = random_select_tiles(p_files, num_tiles = int(n_tiles * 0.33))

    k_files = _setup_bag_region(root_folder, indices_folder, pname, region3)
    k_files = random_select_tiles(k_files, num_tiles = int(n_tiles * 0.33))

    wsi_file = np.concatenate((c_files, p_files, k_files), axis = 0)
    wsi_file= shuffle(wsi_file, random_state = seed)

    return wsi_file

def _combination_4_regions(root_folder, indices_folder, pname, n_tiles = 10000, seed = 2452):
    c_files = _setup_bag_region(root_folder, indices_folder, pname, 'Center')
    c_files = random_select_tiles(c_files, num_tiles = int(n_tiles * 0.25))

    p_files = _setup_bag_region(root_folder, indices_folder, pname, 'Periphery')
    p_files = random_select_tiles(p_files, num_tiles = int(n_tiles * 0.25))

    k_files = _setup_bag_region(root_folder, indices_folder, pname, 'R1')
    k_files = random_select_tiles(k_files, num_tiles = int(n_tiles * 0.25))

    t_files = _setup_bag_region(root_folder, indices_folder, pname, 'TNormal')
    t_files = random_select_tiles(t_files, num_tiles = int(n_tiles * 0.25))

    wsi_file = np.concatenate((c_files, p_files, k_files, t_files), axis = 0)
    wsi_file= shuffle(wsi_file, random_state = seed)

    return wsi_file

def _setup_bag_combination(root_folder, indices_folder, pname, region, n_tiles = 10000): 
    print("Get bag of {}".format(region))

    if region == 'CP':
        return _combination_2_regions(root_folder, indices_folder, pname, "Center", "Periphery", n_tiles = 10000)
    if region == 'CR1':
        return _combination_2_regions(root_folder, indices_folder, pname, "Center", "R1", n_tiles = 10000)
    if region == 'CTNormal':
        return _combination_2_regions(root_folder, indices_folder, pname, "Center", "TNormal", n_tiles = 10000)
    if region == 'PR1':
        return _combination_2_regions(root_folder, indices_folder, pname, "Periphery", "R1", n_tiles = 10000)
    if region == 'PTNormal':
        return _combination_2_regions(root_folder, indices_folder, pname, "Periphery", "TNormal", n_tiles = 10000)
    if region == 'R1TNormal':
        return _combination_2_regions(root_folder, indices_folder, pname, "R1", "TNormal", n_tiles = 10000)
    if region == 'CPR1':
        return _combination_3_regions(root_folder, indices_folder, pname, "Center", "Periphery", "R1", n_tiles = 10000)
    if region == 'CPTNormal':
        return _combination_3_regions(root_folder, indices_folder, pname, "Center", "Periphery", "TNormal", n_tiles = 10000)
    if region == 'CR1TNormal':
        return _combination_3_regions(root_folder, indices_folder, pname, "Center", "R1", "TNormal", n_tiles = 10000)
    if region == 'PR1TNormal':
        return _combination_3_regions(root_folder, indices_folder, pname, "Periphery", "R1", "TNormal", n_tiles = 10000)
    if region == 'All':
        return _combination_4_regions(root_folder, indices_folder, pname, n_tiles = 10000)
    return None

def read_csv(csv_file):
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    patients = {}
    for row in csvreader:
        name, status, year = row
        patients[name] = (np.array([int(status)]), np.array([float(year)]))
    file.close()
    return patients

def read_ckpt(ckpt_file):
    valid_data = torch.load(ckpt_file)
    valid_patients = valid_data['patients']
    valid_labels = valid_data['labels']
    assert(len(valid_patients) == len(valid_labels))
    patients = {}
    for idx in range(len(valid_patients)):
        ptx_name = valid_patients[idx]
        status, year = valid_labels[idx][0], valid_labels[idx][1]
        patients[ptx_name] = (np.array([status]), np.array([year]))
    return patients

def load_patient_dict_and_inference(ckpt_path, root_folder, indices_folder, csv_patient, combination = 'CP'):
    ext = csv_patient.split('.')[-1]
    if ext == 'csv':
       labels_dict = read_csv(csv_patient)
    elif ext == 'ckpt':
       labels_dict = read_ckpt(csv_patient)
    
    y_preds  = inference_validation(ckpt_path, 
                        root_folder, 
                        indices_folder, 
                        labels_dict,
                        comb = combination)
    for p in y_preds:
        print('{}\t{}'.format(p, y_preds[p]))
    
if __name__ == '__main__':
    root = '/media/monc/Disk2/Models/DeepMIL_Survival/lightning_logs_WSI_znormal_region_grade/stratified_split2/default/'
    ckpt_path = root + 'version_2/checkpoints/epoch=25-step=3873.ckpt'
    print(ckpt_path)

    # On bergonie validation
    # testing_patients = '/media/monc/Disk2/Models/DeepMIL_Survival/exports_region_grade/ST_valid/validation_MFS_Grade_CPR1_20230403161636.ckpt'
    # root_folder = '/media/monc/Disk2/Data/Bergonie_features/WSI_features_znormal'
    # indices_folder = '/media/monc/Disk2/Data/Bergonie_features/All_regions_features_znormal'

    # On IGR testing
    testing_patients = '/bergonie_data/Bergonie/csv/survival_data/perisarc/MFS_metas_only.csv'
    root_folder = '/media/monc/Disk2/Data/New_cohort/IGR_PeriSarc_features_znormal'
    indices_folder = '/media/monc/Disk2/Data/New_cohort/IGR_Perisarc_regions'

    load_patient_dict_and_inference(ckpt_path, root_folder, indices_folder, testing_patients, combination = 'CPR1')
    print("Finish !!!!")

