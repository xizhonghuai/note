### GAN网络结构

生成对抗式网络由两个子网络组成，生成网络(Generator，**G**)和判别网络(Discriminator， **D**)。

生成网络G用于生成样本，我们希望生成的样本与真实的样本越接近越好。判别网络D是一个二分类模型，用于区分样本的是**真样本**还是**假样本**(生成器生成的样本),我们希望判别器能够很好的区别样本的真假。

生成器类似自编码器中的解码部分，将隐变量还原成样本数据，这里的隐变量是一个随机噪声，即生成器的输入是随机噪声，输出是样本数据。如何输出期望的样本，是通过网络训练让生成器去学习期望样本的分布。

判别器提供了识别真假样本的能力。真样本与生成器生成的样本(假样本)组成了判别器的训练集。当输入真样本时希望判别器输出为1，输入生成器生成的假样本时判别器输出为0。

生成器通过训练，希望生成的样本能够骗过判别器，使判别器输出为1，到达以假乱真的目的。而判别器也在通过训练提高自己的鉴别真伪的能力。随着模型训练迭代次数增加，生成器与判别器在互相博弈的过程中学习，最终会到达一个均衡点。即生成器能够生成和真实样本非常接近的数据，由于生成器生成的样本与真实样本分布已非常接近，判别器已无法判断真伪，最终输出的为0.5。



<img src="img\1.png" alt="1" style="zoom:80%;" />

### GAN网络损失函数

对于判别器而言，输入包括真样本和假样本数据，设输入真样本时判别器输出real_y，输入假样本时判别器输出为fake_y,则判别器的损失函数可表示为：
$$
d\_loss = -[log(real_y) +log(1-fake_y)]
$$
对于生成器而言，希望生成的图片能够骗过判别器，即希望判别器输出1。设当生成器生成的样本输入，判别器输出为fake_y，则生成器的损失函数表示为：
$$
g\_loss = -log(fake_y)
$$


在训练生成器时，判别器的网络参数应固定，反向传播仅更新生成器参数变量。

在训练判别器时，生成器的网络参数应固定，反向传播仅更新判别器参数变量。



### GAN网络实现

```python
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.datasets.fashion_mnist as fashion_mnist
import tensorflow.python.keras.datasets.mnist as mnist
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

TRAINING_STEPS = 100000
batch_size = 500

# 定义fashion_mnist数据加载初始化
def data_init():
    (x_train, _), (_, _) = fashion_mnist.load_data()
    # (x_train, _), (_, _) = mnist.load_data("E://python//tensorflow_study//MNIST_data//mnist.npz")

    x_train = np.reshape(x_train, (60000, 784))
    x_train = x_train / 255.0
    x_train_np = np.array(x_train[:], dtype='float32')
    return x_train_np

def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False不重复
    if x > n:
        x = n
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

# 随机获取数据
def data_batch_set(input_data):
    data_len = len(input_data)
    index = getRandomIndex(data_len, batch_size)
    return input_data[index]

# 归一化到指定区间[a,b]
def tf_normalize(tensor, a, b):
    tensor_max = tf.reduce_max(tensor)
    tensor_min = tf.reduce_min(tensor)
    return a + (((b - a) / (tensor_max - tensor_min)) * (tensor - tensor_min))


# 归一化到指定区间[a,b]
def np_normalize(np_array, a, b):
    max = np.max(np_array)
    min = np.min(np_array)
    return a + (((b - a) / (max - min)) * (np_array - min))

# 定义层函数
def add_layer(input, in_size, out_size, active_function=None):
    # input 输入矩阵
    # in_size 输入矩阵列大小
    # out_size 输出矩阵列大小
    # active_function 激活函数
    weighs = tf.compat.v1.get_variable('f_weighs', shape=[in_size, out_size],
                                       initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    bais = tf.compat.v1.get_variable('f_bais', shape=[1,out_size]
                                     , initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    # 激励输入
    z_i = tf.matmul(input, weighs)+bais
    if active_function == None:
        return z_i
    return active_function(z_i)


# 定义生成器前向传播
def generator_inference(input_tensor, reuse_variables=False):
    # 输入是batchX128噪声数据
    with tf.name_scope("G"):
        with tf.compat.v1.variable_scope("generator_1", reuse=reuse_variables):
            out_1 = add_layer(input_tensor, 128, 256)
            out_1 = tf.nn.leaky_relu(out_1)
        with tf.compat.v1.variable_scope("generator_2", reuse=reuse_variables):
            out_2 = add_layer(out_1, 256, 28*28)
            out_2 = tf.nn.sigmoid(out_2)

    return out_2


# 定义判别器前向传播
def discriminator_inference(input_tensor, reuse_variables=False):
    # 输入是batchX784图片数据
    with tf.name_scope("D"):
        with tf.compat.v1.variable_scope("discriminator_1", reuse=reuse_variables):
            out_1 = add_layer(input_tensor, 784, 256, tf.nn.leaky_relu)
        with tf.compat.v1.variable_scope("discriminator_2", reuse=reuse_variables):
            out_2 = add_layer(out_1, 256, 1, tf.nn.sigmoid)

    return out_2


# 获取生成器、判别器需要训练的变量
def get_trainable_variables():
    t_vars = tf.compat.v1.trainable_variables()
    g_vars = [var for var in t_vars if 'generator_' in var.name]
    d_vars = [var for var in t_vars if 'discriminator_' in var.name]
    return g_vars, d_vars


# 高斯分布噪声
def create_gaussian_noise(batch_size):
    array = np.array(np.random.normal(loc=0.0, scale=1.0, size=(batch_size, 128)), dtype='float32')
    return array


# 保存测试图片
def save_images(sess, batch_images):
    images_data = []
    batch_images = batch_images * 255.
    image_size = 28
    frame_size = 4

    image_num = len(batch_images)
    if image_num == 1:
        data_temp = 255. * np.ones(shape=(image_size + frame_size, image_size + frame_size))
        data_temp[frame_size:frame_size + image_size, frame_size:frame_size + image_size] = np.reshape(batch_images[0],
                                                                                                       newshape=(
                                                                                                           image_size,
                                                                                                           image_size))
        images_data.append(data_temp)
    else:
        for image in batch_images:
            data_temp = 255. * np.ones(shape=(image_size + frame_size, image_size + frame_size))
            data_temp[frame_size:frame_size + image_size, frame_size:frame_size + image_size] = np.reshape(image,
                                                                                                           newshape=(
                                                                                                               image_size,
                                                                                                               image_size))

            images_data.append(data_temp)

    image_size = image_size + frame_size
    image_np = 255. * np.ones(shape=(image_size, image_num * image_size))
    i = 0
    for image_data in images_data:
        image_np[:, i * image_size:i * image_size + image_size] = image_data
        i = i + 1

    image_np = tf.cast(image_np, dtype=tf.uint8)
    image_np = tf.reshape(image_np, (image_size, image_num * image_size, 1))
    jpg_data = tf.image.encode_jpeg(image_np)
    with gfile.FastGFile("fashion_test_img/tt.jpg", "wb") as f:
        try:
            f.write(sess.run(jpg_data))
            print("---------------------Picture generation succeeded---------------------")
        except Exception:
            print("err")


# 定义模型训练过程
def train():
    global_step_d = tf.Variable(0, trainable=False)
    global_step_g = tf.Variable(0, trainable=False)

    # 判别器输入（真实图片）
    image_x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    # 生成器输入（随机噪声）
    noise_x = tf.compat.v1.placeholder(tf.float32, [None, 128])
    # 真实图片的概率
    real_p = discriminator_inference(input_tensor=image_x)
    # 生成器生成的图片
    fake_images = generator_inference(input_tensor=noise_x)

    # 判断生成器生成图片的概率
    fake_p = discriminator_inference(fake_images, reuse_variables=True)

    # 损失函数
    d_loss = -1. * tf.reduce_mean(tf.compat.v1.log(real_p) + tf.compat.v1.log(1.0 - fake_p))
    g_loss = -1. * tf.reduce_mean(tf.compat.v1.log(fake_p))

    # 获取需要训练的变量
    g_vars, d_vars = get_trainable_variables()
    # 使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数。
    train_step_d = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss=d_loss
                                                                                              ,
                                                                                              global_step=global_step_d
                                                                                              , var_list=d_vars)
    train_step_g = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss=g_loss
                                                                                   , global_step=global_step_g
                                                                                   , var_list=g_vars)
    # 初始化会话并开始训练过程。
    init_var = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_var)
        _ = tf.compat.v1.summary.FileWriter('tensorboard/', sess.graph)
        # 真实样本
        real_images = data_init()
        # 开始交替训练D&G
        for i in range(TRAINING_STEPS):

            train_batch_images = data_batch_set(real_images)
            train_batch_noise = create_gaussian_noise(batch_size)
            _, d_loss_value, real_p_value, fake_p_value = sess.run([train_step_d, d_loss, real_p, fake_p],
                                                                   feed_dict={image_x: train_batch_images,
                                                                              noise_x: train_batch_noise})
            _, g_loss_value = sess.run([train_step_g, g_loss],
                                       feed_dict={noise_x: train_batch_noise})

            if i % 5 == 0:
                # loss
                print("g_loss=%f, d_loss=%f ;[real_p = %f,fake_p = %f] step(%d) " % (
                    g_loss_value, d_loss_value,
                    real_p_value[0], fake_p_value[0], i))

            if i % 20 == 0:
                fake_imgs = generator_inference(create_gaussian_noise(8), reuse_variables=True)
                save_images(sess, sess.run(fake_imgs))


def main():
    train()


if __name__ == '__main__':
    main()
```

<img src="img\2.png" alt="2" style="zoom:75%;" />



###  DCGAN

常规的GAN网络通过全连接实现。DCGAN使用**转置卷积层**实现的生成网络，普通卷积层来实现的判别网络。

```python
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.datasets.fashion_mnist as fashion_mnist
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

TRAINING_STEPS = 10000000000
batch_size = 50


# 定义fashion_mnist数据加载初始化
def data_init():
    (x_train, _), (_, _) = fashion_mnist.load_data()
    # (x_train, _), (_, _) = mnist.load_data("E://python//tensorflow_study//MNIST_data//mnist.npz")
    x_train = (x_train - 127.5) / 127.5
    # 为了演示效果取部分图片
    x_train_np = np.array(x_train[0:200], dtype='float32')
    return np.reshape(x_train_np, newshape=(200, 28, 28, 1))


def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False不重复
    if x > n:
        x = n
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index


# 随机获取数据
def data_batch_set(input_data):
    data_len = len(input_data)
    index = getRandomIndex(data_len, batch_size)
    return input_data[index]


# 归一化到指定区间[a,b]
def np_normalize(np_array, a, b):
    max = np.max(np_array)
    min = np.min(np_array)
    return a + (((b - a) / (max - min)) * (np_array - min))


# 批处理归一化 归一化每个batch数据，避免梯度消失于梯度爆炸问题，解决了训练不稳定的问题
def batch_normalization(value, in_size, training=True):
    '''
    :param value: tensor shape=[b,i,j,c] 或 [b,n]
    :param in_size:输入列大小，对于格式为"NHWC"的数据in_size=C
    :param training:
    :return:
    '''
    # gamma一般初始化为1
    gamma = tf.compat.v1.get_variable("bn_gamma"
                                      , shape=[in_size]
                                      , initializer=tf.compat.v1.ones_initializer)
    # beta一般初始化为0
    beta = tf.compat.v1.get_variable("bn_beta"
                                     , shape=[in_size]
                                     , initializer=tf.compat.v1.zeros_initializer)

    mean = tf.compat.v1.get_variable("bn_mean"
                                     , shape=[in_size]
                                     , initializer=tf.compat.v1.zeros_initializer
                                     , trainable=False)
    variance = tf.compat.v1.get_variable("bn_variance"
                                         , shape=[in_size]
                                         , initializer=tf.compat.v1.zeros_initializer
                                         , trainable=False)
    # 接近零的一个极小值，防止分母为零
    epsilon = 1e-4
    axis = list(range(len(value.get_shape()) - 1))

    def batch_norm_training():
        # 对于卷积网络，确保计算的是每个batch同一特征图上的平均值和方差
        # 对于全连接网络，确保计算的是batch同一神经元输出的平均值与方差
        batch_mean, batch_variance = tf.nn.moments(value, axis)
        # 滑动平均
        decay = 0.10  # 衰减系数
        train_mean = tf.compat.v1.assign(mean, mean * decay + batch_mean * (1 - decay))
        train_variance = tf.compat.v1.assign(variance, variance * decay + batch_variance * (1 - decay))
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(value, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        return tf.nn.batch_normalization(value, mean, variance, beta, gamma, epsilon)

    if training:
        return batch_norm_training()
    return batch_norm_inference()


# 定义全连接层函数
def add_layer(input, in_size, out_size, active_function=None):
    # input 输入矩阵
    # in_size 输入矩阵列大小
    # out_size 输出矩阵列大小
    # active_function 激活函数
    weighs = tf.compat.v1.get_variable('f_weighs', shape=[in_size, out_size],
                                       initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))
    bais = tf.compat.v1.get_variable('f_bais', shape=[1, out_size]
                                     , initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))
    # 激励输入
    z_i = tf.matmul(input, weighs) + bais
    if active_function is None:
        return z_i
    return active_function(z_i)


# 定义池化层函数
def add_pool_layer(input, pool_shape, strides_shape, padding='SAME'):
    # input 输入矩阵（四维张量,(b,i,j,c))
    # pool_shape 池化过滤器参数，四维向量 pool_shape[1:2] 过滤器尺寸
    # pool_shape[0]、pool_shape[3] 必须是1
    # strides_shape 四维向量,卷积步长第1、4维必须是1
    # padding 填充方式,SAME表示添加全0填充“VALID”表示不添加

    # tf.nn.max_pool 实现了最大池化层的前向传播过程，
    # 它的参数和tf.nn.conv2d 函数类似。
    # ksize 提供了过滤器的尺（第一维度与最后维度必须是1）、strides 提供了步长信息（第一维度与最后维度必须是1）， padding 提供填充方式。
    return tf.nn.max_pool(input, ksize=pool_shape, strides=strides_shape, padding=padding)


# 定义卷积层函数
def add_cnn_layer(input, filter_shape, strides_shape, padding, active_function=None):
    # input 输入矩阵（四维张量,(b,i,j,c))
    # filter_shape 过滤器参数四位向量 filter_shape[0:1] 过滤器尺寸，filter_shape[2] input矩阵深度（对应C）
    # filter_shape[3] 过滤器个数
    # strides_shape 四维向量,卷积步长第1、4维必须是1
    # padding 填充方式,SAME表示添加全0填充“VALID”表示不添加
    # active_function 激活函数
    filter_weight = tf.compat.v1.get_variable('c_weighs', shape=filter_shape,
                                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))

    # [filter_shape[3]为过滤器的深度，也是神经网络中下一层节点矩阵的深度。
    biases = tf.compat.v1.get_variable('c_bais', shape=[filter_shape[3]],
                                       initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))

    # tf.nn.conv2d 提供了一个非常方便的函数来实现卷积层前向传播的算法。
    # 第一个参数为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，比如在输入层， 如input[0, :, :, ：］表示第一张图片
    # 第二个参数提供了卷积层的权重，
    # 第三个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求一定是1。
    # 第四个参数是填充（padding)的方法， 其中SAME表示添加全0填充“VALID”表示不添加
    conv = tf.nn.conv2d(input, filter_weight, strides=strides_shape, padding=padding)
    # tf.nn.bias_add 提供了一个方便的函数给每一个节点加上偏置项。
    z_out = tf.nn.bias_add(conv, biases)
    # 将计算结果通过激活函数完成非线性化。

    if active_function is None:
        return z_out
    return active_function(z_out)


# 定义转置卷积层函数
def add_cnn_transpose_layer(input, filter_shape, strides_shape, out_shape, active_function=None, padding='SAME'):
    # input 输入矩阵（四维张量,(b,i,j,c))
    # filter_shape 过滤器参数四位向量 filter_shape[0:1] 过滤器尺寸，filter_shape[2] 过滤器个数,
    # filter_shape[3] input矩阵深度（对应C）
    # strides_shape 四维向量,卷积步长第1、4维必须是1
    # padding 填充方式,SAME表示添加全0填充“VALID”表示不添加
    # out_shape 输出形状,四维张量,(b,i,j,c)（输出形状根据常规卷积运算能够得到输入形状）
    # active_function 激活函数
    filter_weight = tf.compat.v1.get_variable('ct_weighs', shape=filter_shape,
                                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))

    # out_shape[3]为输出通道数，也是神经网络中下一层节点矩阵的深度。
    biases = tf.compat.v1.get_variable('ct_bais', shape=[out_shape[3]],
                                       initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))

    # tf.nn.conv2d 提供了一个非常方便的函数来实现卷积层前向传播的算法。
    # 第一个参数为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，比如在输入层， 如input[0, :, :, ：］表示第一张图片
    # 第二个参数提供了卷积层的权重，
    # 第三个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求一定是1。
    # 第四个参数是填充（padding)的方法， 其中SAME表示添加全0填充“VALID”表示不添加
    conv = tf.nn.conv2d_transpose(input=input
                                  , filters=filter_weight
                                  , strides=strides_shape
                                  , padding=padding
                                  , output_shape=out_shape)
    # tf.nn.bias_add 提供了一个方便的函数给每一个节点加上偏置项。
    z_out = tf.nn.bias_add(conv, biases)
    if active_function is None:
        return z_out

        # 将计算结果通过激活函数完成非线性化。
    return active_function(z_out)


# 定义生成器前向传播
def generator_inference(input_tensor, reuse_variables=False):
    # 输入是[batch,7*7*64]噪声数据
    with tf.name_scope("G"):
        with tf.compat.v1.variable_scope("generator_1", reuse=reuse_variables):
            out_0 = add_layer(input_tensor, 6 * 6 * 128, 6 * 6 * 128, tf.nn.tanh)
            out_1 = tf.reshape(out_0, (batch_size, 6, 6, 128))
        with tf.compat.v1.variable_scope("generator_2", reuse=reuse_variables):
            out_2 = add_cnn_transpose_layer(out_1
                                            , filter_shape=[4, 4, 64, 128]
                                            , strides_shape=[1, 2, 2, 1]
                                            , out_shape=[batch_size, 14, 14, 64]
                                            , active_function=tf.nn.leaky_relu
                                            , padding='VALID')
            out_2 = batch_normalization(out_2, 64, training=not reuse_variables)
        with tf.compat.v1.variable_scope("generator_3", reuse=reuse_variables):
            out_3 = add_cnn_transpose_layer(out_2
                                            , filter_shape=[2, 2, 1, 64]
                                            , strides_shape=[1, 2, 2, 1]
                                            , out_shape=[batch_size, 28, 28, 1]
                                            , active_function=tf.nn.tanh
                                            )
            out_3 = batch_normalization(out_3, 1, training=not reuse_variables)

    return out_3


# 定义判别器前向传播
def discriminator_inference(input_tensor, reuse_variables=False):
    # 输入是batchX784图片数据
    with tf.name_scope("D"):
        with tf.compat.v1.variable_scope("discriminator_0", reuse=reuse_variables):
            out_1 = add_cnn_layer(input=input_tensor
                                  , filter_shape=[5, 5, 1, 64]
                                  , strides_shape=[1, 1, 1, 1]
                                  , padding='SAME'
                                  , active_function=tf.nn.leaky_relu)

            out_1 = add_pool_layer(out_1, pool_shape=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

        with tf.compat.v1.variable_scope("discriminator_1", reuse=reuse_variables):
            out_2 = add_cnn_layer(input=out_1
                                  , filter_shape=[5, 5, 64, 128]
                                  , strides_shape=[1, 1, 1, 1]
                                  , padding='SAME',
                                  active_function=tf.nn.leaky_relu)
            # 7*7*128
            out_2 = add_pool_layer(out_2, pool_shape=[1, 2, 2, 1], strides_shape=[1, 2, 2, 1])

            # 该层的输入是一个向量，需要将第四层7x7x128拉直为一维数组
            out_2_shape = out_2.get_shape().as_list()
            # out_4_shape[O］为一个 batch 中数据的个数。
            nodes = out_2_shape[1] * out_2_shape[2] * out_2_shape[3]
            # 通过 tf.reshape 函数将第四层的输出变成一个 batch 的向量。
            # reshaped_shape = (None,nodes)
            reshaped = tf.reshape(out_2, [batch_size, nodes])

        with tf.compat.v1.variable_scope("discriminator_4", reuse=reuse_variables):
            # 输入(None,nodes)
            # 输出 (None,512)
            out_5 = add_layer(reshaped, nodes, 512, tf.nn.leaky_relu)

        with tf.compat.v1.variable_scope("discriminator_5", reuse=reuse_variables):
            # 第六层全连接层
            # 输入(None,512)
            # 输出 (None,1)
            out_6 = add_layer(out_5, 512, 1, tf.nn.sigmoid)
    return out_6


# 获取生成器、判别器需要训练的变量
def get_trainable_variables():
    t_vars = tf.compat.v1.trainable_variables()
    g_vars = [var for var in t_vars if 'generator_' in var.name]
    d_vars = [var for var in t_vars if 'discriminator_' in var.name]
    return g_vars, d_vars


# 高斯分布噪声
def create_gaussian_noise(batch_size):
    array = np.array(np.random.normal(loc=0.0, scale=0.2, size=(batch_size, 6 * 6 * 128)), dtype='float32')
    return array


# 保存测试图片
def save_images(sess, batch_images):
    images_data = []
    batch_images = batch_images * 255.
    image_size = 28
    frame_size = 4

    image_num = len(batch_images)
    if image_num == 1:
        data_temp = 255. * np.ones(shape=(image_size + frame_size, image_size + frame_size))
        data_temp[frame_size:frame_size + image_size, frame_size:frame_size + image_size] = (batch_images[0])[:, :, 0]
        images_data.append(data_temp)
    else:
        for image in batch_images:
            data_temp = 255. * np.ones(shape=(image_size + frame_size, image_size + frame_size))
            data_temp[frame_size:frame_size + image_size, frame_size:frame_size + image_size] = image[:, :, 0]

            images_data.append(data_temp)

    image_size = image_size + frame_size
    image_np = 255. * np.ones(shape=(image_size, image_num * image_size))
    i = 0
    for image_data in images_data:
        image_np[:, i * image_size:i * image_size + image_size] = image_data
        i = i + 1

    image_np = tf.cast(image_np, dtype=tf.uint8)
    image_np = tf.reshape(image_np, (image_size, image_num * image_size, 1))
    jpg_data = tf.image.encode_jpeg(image_np)
    with gfile.FastGFile("fashion_test_img/tt.jpg", "wb") as f:
        try:
            f.write(sess.run(jpg_data))
            print("---------------------Picture generation succeeded---------------------")
        except Exception:
            print("err")


# 定义模型训练过程
def train():
    global_step_d = tf.Variable(0, trainable=False)
    global_step_g = tf.Variable(0, trainable=False)

    # 判别器输入（真实图片）
    image_x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    # 生成器输入（随机噪声）
    noise_x = tf.compat.v1.placeholder(tf.float32, [None, 6 * 6 * 128])
    # 真实图片的概率
    real_p = discriminator_inference(input_tensor=image_x)
    # 生成器生成的图片
    fake_images = generator_inference(input_tensor=noise_x)

    # 判断生成器生成图片的概率
    fake_p = discriminator_inference(fake_images, reuse_variables=True)

    # 损失函数
    d_loss = -1. * tf.reduce_mean(tf.compat.v1.log(real_p) + tf.compat.v1.log(1.0 - fake_p))
    g_loss = -1. * tf.reduce_mean(tf.compat.v1.log(fake_p))

    # 获取需要训练的变量
    g_vars, d_vars = get_trainable_variables()
    # 使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数。
    train_step_d = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00002).minimize(loss=d_loss
                                                                                    ,
                                                                                    global_step=global_step_d
                                                                                    , var_list=d_vars)
    train_step_g = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00009).minimize(loss=g_loss
                                                                                    , global_step=global_step_g
                                                                                    , var_list=g_vars)

    # 初始化会话并开始训练过程。
    init_var = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_var)

        # _ = tf.compat.v1.summary.FileWriter('tensorboard/', sess.graph)
        # 真实样本
        real_images = data_init()
        # 开始交替训练D&G
        g_loss_value, d_loss_value = 0, 0
        real_p_value, fake_p_value = [], []
        for i in range(TRAINING_STEPS):

            print("-------------train d-------------")
            for k in range(5):
                train_batch_images = data_batch_set(real_images)
                train_batch_noise = create_gaussian_noise(batch_size)
                _, d_loss_value, real_p_value, fake_p_value = sess.run([train_step_d, d_loss, real_p, fake_p],
                                                                       feed_dict={image_x: train_batch_images,
                                                                                  noise_x: train_batch_noise})
                print("g_loss=%f, d_loss=%f ;[real_p = %f,fake_p = %f] step(%d) " % (
                    g_loss_value, d_loss_value,
                    real_p_value[0], fake_p_value[0], i))

            print("-------------train g-------------")
            for k in range(50):
                train_batch_noise = create_gaussian_noise(batch_size)
                _, g_loss_value = sess.run([train_step_g, g_loss],
                                           feed_dict={noise_x: train_batch_noise})
                print("g_loss=%f, d_loss=%f ;[real_p = %f,fake_p = %f] step(%d) " % (
                    g_loss_value, d_loss_value,
                    real_p_value[0], fake_p_value[0], i))



                # loss
            print("g_loss=%f, d_loss=%f ;[real_p = %f,fake_p = %f] step(%d) " % (
                g_loss_value, d_loss_value,
                real_p_value[0], fake_p_value[0], i))
            # print(sess.run(sess.graph.get_tensor_by_name("discriminator_1/bn_gamma:0")))

            if i % 2 == 0:
                fake_imgs = sess.run(generator_inference(create_gaussian_noise(batch_size), reuse_variables=True))
                fake_imgs = np_normalize(fake_imgs[0:4], 0, 1)
                save_images(sess, fake_imgs)


def main():
    train()


if __name__ == '__main__':
    main()

```

### 原始GAN网络存在的问题

* 训练难度大，需要衡量判断别器与生成器各自训练的程度。判别器越好，更新生成器的参数梯度消失越明显，判别器越弱，导致生成器生成的样本无法逼近真实样本分布。

* 超参数敏感。batch、学习速率、参数初始状态、以及网络结构的变动，对训练结果影响很大。甚至无法训练。

* 模式崩塌。生成器生成的样本单一化。

* 没有一个量化指标来评判网络训练进程，只能通过肉眼观察生成样本与真实样本是否接近来确定是否停止训练。

  

### WGAN

WGAN 对原始的GAN网络进行改进，能够有效解决原始GAN网络的一些问题。通过**EM距离**作为目标优化函数，来训练网络。

WGAN具体改进措施:

* 去掉生成器最后一层的sigmod函数
* 生成器与判别器的loss去掉log
* 每次更新判别器的参数前把他们的绝对值限制到某一个常数c
* 不用基于动量(Adam)的优化算法,推荐使用RMSprop、SGD

> 注意：WGAN 仅有效解决原始GAN网络训练过程中的问题，并不一定能提升生成质量，其生成样本的质量与网络结构有很大关系。





  

  
