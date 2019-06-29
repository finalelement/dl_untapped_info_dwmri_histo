import os
from data import data_load, test_data, log_data_load, ln_test_data

# Importing different types of Networks
from models import build_nn_resnet
# Importing Training functions
from models import train_network

from tensorflow.python.client import device_lib
# Just Check if GPU is being used or not
print(device_lib.list_local_devices())

ln_ip_data_path = r'\python_code\ln_train_2m_b3k_input_8.mat'
ln_op_data_path = r'\python_code\ln_train_2m_b3k_output_8.mat'

#ip_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_input_2m.mat'
#op_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_output_2m.mat'

#test_data_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\python_code\b3k_test_input_1m.mat'
ln_test_data_path = r'\python_code\ln_test_1m_b3k_input.mat'

# Loading Data
X, y = log_data_load(ln_ip_data_path, ln_op_data_path)

# Reduce y to 8th order
y = y[:,:45]

print('Data Loaded ... \n')

res_model = build_nn_resnet()

print('Network Constructed ... \n')
print('Training Network ... \n')

res_model = train_network(res_model, X, y, num_epoch=500, batch=1000)

print('Making Predictions and Saving file')

save_file_path = r'seq_ln_resnet.mat'
ln_test_data(res_model, ln_test_data_path, save_file_path)








