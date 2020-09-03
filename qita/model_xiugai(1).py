from keras.models import Model
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, Lambda, Concatenate, \
    Multiply, RepeatVector, Permute, Flatten, Activation
import keras.backend as K
from keras import optimizers
from selfDef import myLossFunc, Attention, coAttention_para, zero_padding, tagOffSet
import pickle
import h5py
import numpy as np
import keras
from keras_gcn import GraphConv
# 定义参数
num_tags = 3896
num_words = 212000
index_from_text = 3
index_from_tag = 2
seq_length = 30    # max length of text sequence
batch_size = 512
embedding_size = 300
attention_size = 200
dim_k = 100
num_region = 7*7
drop_rate = 0.75
maxTagLen = 48    # max length of tag sequence
num_epoch = 30
numHist = 2    # historical posts number for each user
numTestInst = 64264    # if you're going to use predict_generator, modify this parameter as your testSet size.
testBatchSize = 40
print("Experiment parameters:")
# 打印嵌入维度和训练批次
print("embedding_size: %d, num_epoch: %d" % (embedding_size, num_epoch))


# 定义模型
def modelDef():
    '''
    原始是八个是输入，现在要加入两个输入(X,A),标签的特征X和邻接矩阵A

    :return:
    '''
    # 当前的文本和图像特征
    # 输入的图片特征（7,7,512）
    inputs_img = Input(shape=(7, 7, 512))
    # 输出的文本大小（30,）
    inputs_text = Input(shape=(seq_length,))

    # 历史的文本和图像特征
    # 历史图片0（7,7,512）
    inputs_hist_img_0 = Input(shape=(7, 7, 512))
    # 历史图片1（7,7,512）
    inputs_hist_img_1 = Input(shape=(7, 7, 512))
    # 历史文本大小(30,)
    inputs_hist_text_0 = Input(shape=(seq_length,))
    inputs_hist_text_1 = Input(shape=(seq_length,))
    # 历史标签大小(48,)
    inputs_hist_tag_0 = Input(shape=(maxTagLen,))
    inputs_hist_tag_1 = Input(shape=(maxTagLen,))


    # hashtag输入的特征和邻接矩阵
    inputs_hashtag_feature = Input(shape=(3896,3896))
    inputs_hashtag_adj = Input(shape=(3896,3896))




    # content Modeling块
    # 当前帖子的特征处理
    # shared layers,定义共享层的参数
    # 标签嵌入层  词汇表大小（21200+2），词向量维度300，输入序列长度（40，
    tagEmbeddings = Embedding(input_dim=num_words + index_from_tag, output_dim=embedding_size,
                              mask_zero=True, input_length=maxTagLen)
    # 文本嵌入层  词汇表大小（21200+3），词向量维度300，输入序列长度（30，
    textEmbeddings = Embedding(input_dim=num_words + index_from_text, output_dim=embedding_size,
                               mask_zero=True, input_length=seq_length)
    # 利用LSTM处理，输入的是嵌入大小（300）
    lstm = LSTM(units=embedding_size, return_sequences=True)
    # 进过一个全连接层处理
    dense = Dense(embedding_size, activation="tanh", use_bias=False)
    # 将图片特征转换为49*512的维度的
    reshape = Reshape(target_shape=(num_region, 512))
    # 调用co-attention进行处理
    coAtt_layer = coAttention_para(dim_k=dim_k)
    tag_att = Attention(attention_size)





    # query post representation
    # 训练帖子表示
    # 经过嵌入层将文本嵌入成
    text_embeddings = textEmbeddings(inputs_text)
    # 利用lstm对文本进行处理
    tFeature = lstm(text_embeddings)
    # 图片特征处理成49*512维度的
    iFeature = reshape(inputs_img)
    # 经过一个全连接层变成了49*300维度
    iFeature = dense(iFeature)
    # 调用co-attention层，返回融合后的特征（49,300）
    co_feature = coAtt_layer([tFeature, iFeature])

    # 利用两次gcn对tag的图进行处理，形成新的特征
    hashtag_gcn = GraphConv(units=1024, activation='relu')([inputs_hashtag_feature, inputs_hashtag_adj])
    hashtag_gcn1 = Dropout(0.2)(hashtag_gcn)
    hashtag_feature = GraphConv(units=300, activation='softmax')([hashtag_gcn1, inputs_hashtag_adj])

    # 利用处理后的特征进行co-attention处理（iFeature,tFeature）
    # (3896,300 )( 49,300) -> 300
    hashtag_img_co_feature = coAtt_layer([hashtag_feature,iFeature])
    # (3896,300) (30,300) -> 300
    hashtag_text_co_feature = coAtt_layer([hashtag_feature,tFeature])

    # 历史帖子的特征处理（habit Modeling块）
    # historical posts representation
    # 对tag求取注意力
    hist_tag_0 = tag_att(tagEmbeddings(inputs_hist_tag_0))
    # 文本特征嵌入
    hist_text_0 = textEmbeddings(inputs_hist_text_0)
    # 利用lstm处理历史帖子文本特征
    hist_tfeature_0 = lstm(hist_text_0)
    # 将图片特征转换为49*512
    hist_ifeature_0 = reshape(inputs_hist_img_0)
    # 转换为49*300
    hist_ifeature_0 = dense(hist_ifeature_0)
    # 经过一个co-attention获得历史帖子的融合特征
    hist_cofeature_0 = coAtt_layer([hist_tfeature_0, hist_ifeature_0])


    # 同上述处理一样
    #
    hist_tag_1 = tag_att(tagEmbeddings(inputs_hist_tag_1))
    hist_text_1 = textEmbeddings(inputs_hist_text_1)
    hist_tfeature_1 = lstm(hist_text_1)
    hist_ifeature_1 = reshape(inputs_hist_img_1)
    hist_ifeature_1 = dense(hist_ifeature_1)
    hist_cofeature_1 = coAtt_layer([hist_tfeature_1, hist_ifeature_1])



    # 历史帖子找与当前帖子的相似度（habit Modeling块）
    #  si的值，历史特征和当前特征点乘
    sim_0 = Multiply()([hist_cofeature_0, co_feature])
    #
    sim_0 = RepeatVector(1)(Concatenate()([sim_0, hist_tag_0]))
    # 另一个历史特征和当前特征点乘
    sim_1 = Multiply()([hist_cofeature_1, co_feature])
    # 重复一次，将sim值和起那么获得注意力后的标签融合
    sim_1 = RepeatVector(1)(Concatenate()([sim_1, hist_tag_1]))

    sims = Concatenate(axis=1)([sim_0, sim_1])

    # 求取注意力值（历史帖子的）
    # 经过一个tanh函数
    attention = Dense(1, activation='tanh')(sims)
    # 展平
    attention = Flatten()(attention)
    # softmax函数
    attention = Activation('softmax')(attention)
    # 重复3*embedding_size次
    attention = RepeatVector(2 * embedding_size)(attention)
    # 转换维度位置（）
    attention = Permute([2, 1])(attention)

    # 影响t经过attention后的标签*历史帖子的注意力
    influence = Multiply()([sims, attention])
    # 加权求和
    influence = Lambda(lambda x: K.sum(x, axis=1))(influence)
    # 经过一个全连接层转换维度
    influence = Dense(embedding_size)(influence)

    # 历史帖子和当前帖子的特征融合（prediction块）

    h = Concatenate()([co_feature, influence,hashtag_img_co_feature,hashtag_text_co_feature])
    # 经过dropout层
    dropout = Dropout(drop_rate)(h)
    # 全连接层
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)


    # 最终模型建立
    model = Model(inputs=[inputs_img, inputs_text, inputs_hist_img_0, inputs_hist_text_0, inputs_hist_tag_0,
                          inputs_hist_img_1, inputs_hist_text_1, inputs_hist_tag_1,inputs_hashtag_feature,inputs_hashtag_adj],
                  outputs=[Softmax])
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
modelDef()