import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import utils
from modules import Dice


# 构建基本 base 模型，用于对照
class BaseModel(tf.keras.Model):

    def __init__(self, feature_columns, cate_list, behavior_feature_list, units=(80, 40), activation='prelu', maxlen=40,
                 dropout=0., embed_reg=1e-4):

        super(BaseModel, self).__init__()

        self.maxlen = maxlen

        self.cate_list = cate_list

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.seq_len = len(behavior_feature_list)

        # 构建embedding层
        self.sparse_embedding_layers = [layers.Embedding(input_dim=63001,
                                                         output_dim=3,
                                                         embeddings_initializer='random_uniform',
                                                         embeddings_regularizer=l2(embed_reg))
                                        for feat in self.sparse_feature_columns if
                                        feat['feat'] not in behavior_feature_list]

        self.dense_embedding_layers = [layers.Embedding(input_dim=63001,
                                                        output_dim=3,
                                                        embeddings_initializer='random_uniform',
                                                        embeddings_regularizer=l2(embed_reg))
                                       for feat in self.sparse_feature_columns if
                                       feat['feat'] in behavior_feature_list]

        #self.item_embed = layers.Embedding(input_dim=)

        # bn
        self.bn = layers.BatchNormalization(trainable=True)

        # dnn
        self.dnn = [layers.Dense(units=unit, activation=layers.PReLU() if activation == 'prelu' else Dice())
                    for unit in units]

        # dropout
        self.dropout = layers.Dropout(rate=dropout)

        # final
        self.final_output = layers.Dense(1)

    # def concat_embed(self, item):
    #     """
    #     concat the item embedding and categories embedding
    #     :param item: item ID
    #     :return: concated embedding
    #     """
    #
    #     cate = tf.gather(self.cate_list, item)
    #     cate = tf.squeeze(cate, 1) if cate.shape[-1] == 1 else cate
    #     item_embed = self.

    def call(self, inputs):

        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs

        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.sparse_embedding_layers[i](sparse_inputs[:, :, i])], axis=-1)
        seq_embed = tf.concat([self.dense_embedding_layers[i](seq_inputs[:,  i]) for i in range(self.seq_len)], axis=-1)
        item_embed = tf.concat([self.dense_embedding_layers[i](item_inputs[:, i]) for i in range(self.seq_len)], axis=-1)

        print("seq/item embed done!")
        # pooling
        user_info = tf.reduce_sum(seq_embed, axis=1)
        print("pooling done!")

        # concat user_info, cantidate item embedding other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)

        # bn
        info_all = self.bn(info_all)

        # dnn
        for dense in self.dnn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.final_output(info_all))

        print("build BaseModel done!")

        return outputs


if __name__ == "__main__":
    model = BaseModel()
