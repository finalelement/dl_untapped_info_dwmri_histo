import os
from data import data_load, test_data

# Importing different types of Networks
from models import build_nn_resnet
# Importing Training functions
from models import train_network

from tensorflow.python.client import device_lib
# Just Check if GPU is being used or not
print(device_lib.list_local_devices())

ln_ip_data_path = r'\python_code\ln_train_2m_b3k_input.mat'
ln_op_data_path = r'\python_code\ln_train_2m_b3k_output.mat'

ip_data_path = r'\python_code\b3k_input_2m.mat'
op_data_path = r'\python_code\b3k_output_2m.mat'
test_data_path = r'\python_code\b3k_test_input_1m.mat'

model_save_path = r'model_weights.h5'
# Loading Data
X, y = data_load(ip_data_path, op_data_path)

# Reduce y to 8th order
y = y[:,:45]

print('Data Loaded ... \n')

res_model = build_nn_resnet()

print('Network Constructed ... \n')
print('Training Network ... \n')

res_model = train_network(res_model, X, y, num_epoch=400, batch=1000, save_path=model_save_path)

print('Making Predictions and Saving file')

#save_file_path = r'D:\Users\Vishwesh\PycharmProjects\Deep_PNAS\Model_Results_2019\seq_resnet_v2.mat'
save_file_path = r'test_resnet_v2.mat'
test_data(res_model, test_data_path, save_file_path)








