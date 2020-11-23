import pandas as pd
import tensorflow as tf
import numpy as np

# 获取 embedding 特征列表
def get_embedding_features_list():
    
    embedding_features_list = ["cate_id", "brand", 
                                "cms_segid", "cms_group_id", "final_gender_code", "age_level",
                                "pvalue_level", "shopping_level", "occupation", "new_user_class_level"]
    return embedding_features_list

# behavior_log 数据 - 用户 behavior 列表
def get_user_behavior_features():

    user_behavior_features = ["cate","brand"]
    return user_behavior_features

def get_embedding_count(feature, embedding_count):
    return embedding_count[feature].values[0]

# 初步判断为embedding的维度？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
def get_embedding_count_dict(embedding_features_list, embedding_count):
    embedding_count_dict = dict()
    for feature in embedding_features_list:
        embedding_count_dict[feature] = get_embedding_count(feature, embedding_count)
    embedding_count_dict['brand'] = 500000
    embedding_count_dict['cate_id'] = 501578
    embedding_count_dict['final_gender_code'] = 3
    embedding_count_dict['pvalue_level'] = 10
    embedding_count_dict['shopping_level'] = 4
    embedding_count_dict['occupation'] = 5
    embedding_count_dict['new_user_class_level'] = 5
    return embedding_count_dict

# 设定每个 特征的embedding向量长度为64，即64个dense数表示一个特征
def get_embedding_dim_dict(embedding_features_list):
    embedding_dim_dict = dict()
    for feature in embedding_features_list:
        embedding_dim_dict[feature] = 64
    return embedding_dim_dict

def get_data():
    train_data = pd.read_csv("./data/train.csv", sep='\t')
    train_data = train_data.fillna(0)
    # 剔除脏数据，即：无点击列别和品牌的数据，因为也无法训练
    train_data = train_data[train_data['train_data.click_cate'] != 0]
    train_data = train_data[train_data['click_brand'] != 0]
    
    test_data = pd.read_csv("./data/test_csv", sep='\t')
    test_data = test_data.fillna(0)
    test_data = test_data[test_data['click_cate'] != 0]
    test_data = test_data[test_data['click_brand'] != 0]

    embedding_count = pd.read_csv("./data/embedding_count.csv")
    return train_data, test_data, embedding_count

def get_normal_data(data, col):
    return data[col].values

def get_sequence_data(data, col):
    rst = []
    max_length = 0
    for i in data[col].values:
        # 获得序列化的数值，将每一列的的数据用逗号分割，判断数据长度，可以理解为样本个数
        temp = len(list(map(eval, i[1:-1].split(','))))
        if temp > max_length:
            max_length = temp

    for i in data[col].values:
        # temp = [1, 2, 3]
        temp = list(map(eval, i[1:-1].split(',')))
        # max_length = 5, padding = [0.0, 0.0]
        padding = np.zeros(max_length - len(temp))
        # list(np.append(np.array(temp), padding)) -> [1.0, 2.0, 3.0, 0.0, 0.0]
        rst.append(list(np.append(np.array(temp), padding)))
    return rst

def get_length(data, col):
    rst = []
    for i in data[col].values:
        temp = len(list(map(eval, i[1:-1].split(','))))
        rst.append(temp)
    return rst

# 将  xx数据转换成 tensor
def convert_tensor(data):
    return tf.convert_to_tensor(data)

# 获取 train_data，按照batch的形式
def get_batch_data(data, min_batch, batch=100):
    # batch_data = None
    # if min_batch + batch <= len(data):
    #     batch_data = data.loc[min_batch:min_batch + batch - 1]
    # else:
    #     batch_data = data.loc[min_batch:]

    # 随机抽样 batch 个子集
    batch_data = data.sample(n=batch)

    click = get_normal_data(batch_data, 'click')
    target_cate = get_normal_data(batch_data, 'cate_id')
    target_brand = get_batch_data(batch_data, 'brand')
    cms_segid = get_normal_data(batch_data, 'cms_segid')
    cms_group = get_normal_data(batch_data, 'cms_group_id')
    gender = get_normal_data(batch_data, 'final_gender_code')
    age = get_normal_data(batch_data, 'age_level')
    pvalue = get_normal_data(batch_data, 'pvalue_level')
    shopping = get_normal_data(batch_data, 'shopping_level')
    occupation = get_normal_data(batch_data, 'occupation')
    user_class_level = get_normal_data(batch_data, 'new_user_class_level')

    # 
    hist_brand_behavior_click = get_sequence_data(batch_data, 'click_brand')
    hist_brand_behavior_show = get_sequence_data(batch_data, 'show_brand')
    hist_cate_behavior_click = get_sequence_data(batch_data, 'click_cate')
    hist_cate_behavior_show = get_sequence_data(batch_data, 'show_cate')

    click_length = get_length(batch_data, 'click_brand')
    show_length = get_length(batch_data, 'show_brand')

    return tf.one_hot(click, 2), convert_tensor(target_cate), convert_tensor(target_brand), \
        convert_tensor(cms_segid), convert_tensor(cms_group), convert_tensor(gender), \
            convert_tensor(age), convert_tensor(pvalue), convert_tensor(shopping),\
                convert_tensor(occupation), convert_tensor(user_class_level), \
                    convert_tensor(hist_brand_behavior_click), convert_tensor(hist_brand_behavior_show), \
                        convert_tensor(hist_cate_behavior_click), convert_tensor(hist_cate_behavior_show), \
                            min_batch + batch, click_length, show_length

def get_test_data(data):
    batch_data = data.head(150)
    click = get_normal_data(batch_data, 'click')
    target_cate = get_normal_data(batch_data, 'cate_id')
    target_brand = get_batch_data(batch_data, 'brand')
    cms_segid = get_normal_data(batch_data, 'cms_segid')
    cms_group = get_normal_data(batch_data, 'cms_group_id')
    gender = get_normal_data(batch_data, 'final_gender_code')
    age = get_normal_data(batch_data, 'age_level')
    pvalue = get_normal_data(batch_data, 'pvalue_level')
    shopping = get_normal_data(batch_data, 'shopping_level')
    occupation = get_normal_data(batch_data, 'occupation')
    user_class_level = get_normal_data(batch_data, 'new_user_class_level')

    # 
    hist_brand_behavior_click = get_sequence_data(batch_data, 'click_brand')
    hist_brand_behavior_show = get_sequence_data(batch_data, 'show_brand')
    hist_cate_behavior_click = get_sequence_data(batch_data, 'click_cate')
    hist_cate_behavior_show = get_sequence_data(batch_data, 'show_cate')

    click_length = get_length(batch_data, 'click_brand')
    show_length = get_length(batch_data, 'show_brand')

    return tf.one_hot(click, 2), convert_tensor(target_cate), convert_tensor(target_brand), \
        convert_tensor(cms_segid), convert_tensor(cms_group), convert_tensor(gender), \
            convert_tensor(age), convert_tensor(pvalue), convert_tensor(shopping),\
                convert_tensor(occupation), convert_tensor(user_class_level), \
                    convert_tensor(hist_brand_behavior_click), convert_tensor(hist_brand_behavior_show), \
                        convert_tensor(hist_cate_behavior_click), convert_tensor(hist_cate_behavior_show), \
                            click_length, show_length