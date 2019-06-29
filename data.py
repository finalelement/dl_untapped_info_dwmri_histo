import os
from scipy.io import loadmat,savemat
from deep_pnas_py_src.utils import split_X_8th_order

def data_load(ip_path, op_path):

    ip_path = os.path.normpath(ip_path)
    op_path = os.path.normpath(op_path)

    ip_data = loadmat(ip_path)
    op_data = loadmat(op_path)

    X = ip_data['b3k_input_2m']
    y = op_data['b3k_output_2m']

    return X,y

def log_data_load(ip_path, op_path):
    ip_path = os.path.normpath(ip_path)
    op_path = os.path.normpath(op_path)

    ip_data = loadmat(ip_path)
    op_data = loadmat(op_path)

    X = ip_data['sh_ln_dwmri']
    y = op_data['sh_ln_fod']

    return X, y

    #ip_a_name = []
    #op_a_name = []
    #for key, val in ip_data.items():
    #    ip_a_name.append(key)
    #    print(key)

    #for key, val in op_data.items():
    #    op_a_name.append(key)
    #    print(key)

def test_data(model_t, test_data_path, pred_save_path):

    test_data_path = os.path.normpath(test_data_path)
    pred_save_path = os.path.normpath(pred_save_path)

    test_ip = loadmat(test_data_path)
    test_data = test_ip['b3k_test_input_1m']
    preds = model_t.predict(test_data)

    savemat(pred_save_path, mdict={'out_pred': preds})


def ln_test_data(model_t, test_data_path, pred_save_path):

    test_data_path = os.path.normpath(test_data_path)
    pred_save_path = os.path.normpath(pred_save_path)

    test_ip = loadmat(test_data_path)
    test_data = test_ip['sh_ln_dwmri']
    preds = model_t.predict(test_data)

    savemat(pred_save_path, mdict={'out_pred': preds})

def test_refnet_data(model_t, test_data_path, pred_save_path):

    test_data_path = os.path.normpath(test_data_path)
    pred_save_path = os.path.normpath(pred_save_path)

    test_ip = loadmat(test_data_path)
    test_data = test_ip['b3k_test_input_1m']

    x1, x2, x3, x4, x5 = split_X_8th_order(test_data)

    preds = model_t.predict([x1,x2,x3,x4,x5])
    savemat(pred_save_path, mdict={'out_pred': preds})