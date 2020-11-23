import time
import tensorflow as tf
import os


def get_file_name():
    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    return "loss.csv." + now_time


def make_train_loss_dir(file_name, cols=['train_aux_loss', 'train_target_loss', 'train_final_loss'], model='din'):
    f = open("./loss/" + model + "/train_" + file_name, 'w')
    f.write(",".join(cols) + '\n')
    f.close()


def make_test_loss_dir(file_name, cols=['test_aux_loss', 'test_target_loss', 'test_final_loss'], model='din'):
    f = open("./loss/" + model + "/test_" + file_name, 'w')
    f.write(",".join(cols) + '\n')
    f.close()


def add_loss(loss_dict, file_name, cols=['aux_loss', 'target_loss', 'final_loss'], level='train', model='din'):
    loss_list = list()
    for col in cols:
        loss_list.append(loss_dict[col])
    f = open("./loss/" + model + "/level_" + file_name, 'a')
    f.write(",".join(loss_list) + '\n')
    f.close()


def get_input_dim(embedding_dim_dict, user_behavior_features):
    rst = 0
    for feature in user_behavior_features:
        rst += embedding_dim_dict[feature]
    return rst


def concat_features(feature_data_dict):
    concat_list = []
    for k in feature_data_dict:
        concat_list.append(feature_data_dict[k])
    return tf.concat(concat_list, axis=-1)


def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return 0
    except:
        return 1