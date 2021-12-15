YouTube推荐算法
==============


# Deep Neural Networks for YouTube Recommendations

## 论文解读

https://zhuanlan.zhihu.com/p/25343518

https://zhuanlan.zhihu.com/p/24339183?refer=deeplearning-surfing

http://zangbo.me/2017/06/06/Word2Vec_Negative-Sample/

https://blog.csdn.net/yujianmin1990/article/details/80640964

https://www.hardikp.com/2017/09/17/youtube-recommendations/


## 算法实现

### 数据
> 数据下载来自FaceBook的fastText官方数据，具体见：https://github.com/facebookresearch/fastText/blob/master/classification-example.sh ，数据下载及处理如下：

```shell
wget https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz

tar -xzvf dbpedia_csv.tar.gz

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " "}

cat dbpedia_csv/train.csv | normalize_text > dbpedia.train

cat dbpedia_csv/test.csv | normalize_text > dbpedia.test

# 打乱文件顺序
shuf dbpedia.train -o dbpedia.train2

shuf dbpedia.test -o dbpedia.test2
```

### 代码

> 算法参考https://github.com/ogerhsou/Youtube-Recommendation-Tensorflow ，对其进行了详细注释，代码如下：

```python
####################### load packages #####################
import linecache
import numpy as np
import tensorflow as tf
import math

####################### dataset #####################
train_file = "dbpedia_csv/dbpedia.train2"
test_file = "dbpedia_csv/dbpedia.test2"

######## label字典和词袋字典 ########
label_dict = {}
sku_dict = {}

####################### 初始化数据 #####################
def init_data(read_file):
    '''
    获取label词典和词袋：label按照顺序编了个号，作为模拟的时长；词袋是训练数据中所有单词的集合
    '''

    label_cnt = 0
    sku_cnt = 1

    ######## 读取文件 ########
    f = open(read_file, 'r')
    for line in f:
        line = line.strip().split(' ')

        for i in line:

            ##### 获取label #####
            if i.find('__label__') == 0:
                if i not in label_dict:
                    label_dict[i] = label_cnt
                    label_cnt += 1

            ##### 获取单词放入词袋 #####
            else:
                if i not in sku_dict:
                    sku_dict[i] = sku_cnt
                    sku_cnt += 1

######## 获取label词典和词袋 ########
init_data(train_file)


####################### 相关参数设置 #####################
######## 句子窗口长度 ########
max_window_size = 1000
emb_mask = tf.placeholder(tf.float32, shape=[None, max_window_size, 1])

######## embedding size 词向量维度 ########
emb_size = 128
embedding = {
    'input': tf.Variable(tf.random_uniform([len(sku_dict)+1, emb_size], -1.0, 1.0))
    # 'output': tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}


######## x,y placeholder ########
x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])
y_batch = tf.placeholder(tf.int64, [None, 1])

######## embedding placeholder ########
#### 单词个数placeholder ####
word_num = tf.placeholder(tf.float32, shape=[None, 1])

#### 输入的embedding ####
input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)

#### 句子的平均embedding ####
project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding, emb_mask), 1), word_num)


######## 网络相关参数 ########
batch_size = 500
learning_rate = 0.01

training_epochs = 10
display_step = 10

n_classes = 14
n_hidden_1 = 128  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features

#### weights and biases ####
weights = {
    'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]))
}


####################### 读取数据 #####################
def read_data(pos, batch_size, data_lst):
    '''
    :param pos:         数据起始位置
    :param batch_size:  batch size
    :param data_lst:    数据流
    :return:            batch size的数据
    获取batch size的数据
    '''

    ######## 获取batch size数据 ########
    batch = data_lst[pos:pos + batch_size]

    ######## 单词id的存放array ########
    x = np.zeros((batch_size, max_window_size))

    ######## 句子embedding的mask ########
    mask = np.zeros((batch_size, max_window_size))

    ######## label存放的list ########
    y = []

    ######## 读取batch size数据 ########
    word_num = np.zeros((batch_size))

    line_no = 0

    ######## 读取batch size数据 ########
    for line in batch:
        line = line.strip().split(' ')
        col_no = 0

        ###### 获取label即模拟的时长 ######
        y.append(label_dict[line[0]])

        for i in line[1:]:

            ##### 获取单词在词袋中的id，并更新mask #####
            if i in sku_dict:
                x[line_no][col_no] = sku_dict[i]
                mask[line_no][col_no] = 1
                col_no += 1

            ##### 句子长度不能超过设定的长度 #####
            if col_no >= max_window_size:
                break

        word_num[line_no] = col_no
        line_no += 1

    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, max_window_size, 1), word_num.reshape(batch_size, 1)


####################### 定义模型 #####################
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    return layer_2


train_lst = linecache.getlines(train_file)

########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred = multilayer_perceptron(project_embedding, weights, biases)

#### loss 损失计算 ####
#### NCE weights ####
nce_weights = tf.Variable(
    tf.truncated_normal([n_classes, n_hidden_2], stddev=1.0 / math.sqrt(n_hidden_2)))

nce_biases = tf.Variable(tf.zeros([n_classes]))

cost = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=y_batch, inputs=pred,
                                     num_sampled=10, num_classes=n_classes))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases
correct_pred = tf.equal(tf.argmax(out_layer, 1), y_batch)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


##################### train and evaluate model ##########################
########## initialize variables ##########
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    total_batch = int(len(train_lst) / batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            step += 1

            x, y, batch_mask, word_number = read_data(i * batch_size, batch_size, train_lst)

            sess.run(optimizer, feed_dict={x_batch: x, emb_mask: batch_mask, word_num: word_number, y_batch: y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x_batch: x, emb_mask: batch_mask,
                                                                  word_num: word_number, y_batch: y})

                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    test_lst = linecache.getlines(test_file)
    total_batch = int(len(test_lst) / batch_size)

    for i in range(total_batch):
        x, y, batch_mask, word_number = read_data(i*batch_size, batch_size, test_lst)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x_batch: x, y_batch: y, emb_mask: batch_mask,
                                                                 word_num: word_number}))

```




