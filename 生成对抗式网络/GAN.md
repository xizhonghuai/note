### GAN网络结构

生成对抗式网络由两个子网络组成，生成网络(Generator，**G**)和判别网络(Discriminator， **D**)。

生成网络G用于生成样本，我们希望生成的样本与真实的样本越接近越好。判别网络D是一个二分类模型，用于区分样本的是**真样本**还是**假样本**(生成器生成的样本),我们希望判别器能够很好的区别样本的真假。

生成器类似自编码器中的解码部分，将隐变量还原成样本数据，这里的隐变量是一个随机噪声，即生成器的输入是随机噪声，输出是样本数据。如何输出期望的样本，是通过网络训练让生成器去学习期望样本的分布。

判别器提供了识别真假样本的能力。真样本与生成器生成的样本(假样本)组成了判别器的训练集。当输入真样本时判别器输出概率为1，输入生成器生成的假样本时判别器输出概率为0。

生成器通过训练，希望生成的样本能够骗过判别器，使判别器输出概率为1，到达以假乱真的目的。而判别器也在通过训练提高自己的鉴别真伪的能力。随着模型训练迭代次数增加，生成器与判别器在互相博弈的过程中学习，最终会到达一个均衡点。即生成器能够生成和真实样本非常接近的数据，由于生成器生成的样本与真实样本分布已非常接近，判别器已无法判断真伪，最终输出的概率为0.5。



<img src="img\1.png" alt="1" style="zoom:80%;" />

### GAN训练方式



### fashion_mnist数据重建实例

```python
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.datasets.fashion_mnist as fashion_mnist
import tensorflow.python.keras.datasets.mnist as mnist
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

TRAINING_STEPS = 10000
batch_size = 20


# 定义fashion_mnist数据加载初始化
def data_init():
    # (x_train, _), (_, _) = fashion_mnist.load_data()
    (x_train, _), (_, _) = mnist.load_data("E://python//tensorflow_study//MNIST_data//mnist.npz")

    x_train = np.reshape(x_train, (60000, 784))
    x_train = x_train / 255.0
    # 为了演示效果取部分图片
    x_train_np = np.array(x_train[0:batch_size * 3], dtype='float32')
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


# 定义层函数
def add_layer(input, in_size, out_size, active_function=None):
    # input 输入矩阵
    # in_size 输入矩阵列大小
    # out_size 输出矩阵列大小
    # active_function 激活函数
    weighs = tf.compat.v1.get_variable('f_weighs', shape=[in_size, out_size],
                                       initializer=tf.compat.v1.truncated_normal_initializer())
    bais = tf.compat.v1.get_variable('f_bais', shape=[1, out_size]
                                     , initializer=tf.compat.v1.truncated_normal_initializer())
    # 激励输入
    z_i = tf.matmul(input, weighs) + bais
    z_i = tf.nn.dropout(z_i, 0.5)
    if active_function == None:
        return z_i
    return active_function(z_i)


# 定义生成器前向传播
def generator_inference(input_tensor, reuse_variables=False):
    # 输入是batchX128噪声数据
    with tf.name_scope("G"):
        with tf.compat.v1.variable_scope("generator_1", reuse=reuse_variables):
            out_1 = add_layer(input_tensor, 128, 256)
            out_1 = tf.nn.l2_normalize(out_1, dim=1, epsilon=1e-10)
            out_1 = tf.nn.tanh(out_1)
        with tf.compat.v1.variable_scope("generator_2", reuse=reuse_variables):
            out_2 = add_layer(out_1, 256, 512)
            out_2 = tf.nn.l2_normalize(out_2, dim=1, epsilon=1e-10)
            out_2 = tf.nn.tanh(out_2)
        with tf.compat.v1.variable_scope("generator_3", reuse=reuse_variables):
            out_3 = add_layer(out_2, 512, 28 * 28 * 1)
            # out_3 = tf.nn.l2_normalize(out_3, dim=1, epsilon=1e-10)
            out_3 = tf_normalize(out_3, -2.0, 2.0)
            out_3 = tf.nn.sigmoid(out_3)

    return out_3


# 定义判别器前向传播
def discriminator_inference(input_tensor, reuse_variables=False):
    # 输入是batchX784图片数据
    with tf.name_scope("D"):
        with tf.compat.v1.variable_scope("discriminator_1", reuse=reuse_variables):
            out_1 = add_layer(input_tensor, 784, 128, tf.nn.leaky_relu)
        with tf.compat.v1.variable_scope("discriminator_2", reuse=reuse_variables):
            out_2 = add_layer(out_1, 128, 2, tf.nn.leaky_relu)
        # with tf.compat.v1.variable_scope("discriminator_3", reuse=reuse_variables):
        #     out_3 = add_layer(out_2, 128, 2, tf.nn.leaky_relu)

    return out_2


# 获取生成器、判别器需要训练的变量
def get_trainable_variables():
    t_vars = tf.compat.v1.trainable_variables()
    g_vars = [var for var in t_vars if 'generator_' in var.name]
    d_vars = [var for var in t_vars if 'discriminator_' in var.name]
    return g_vars, d_vars


# 随机均匀分布噪声
def create_noise(batch):
    return np.array(np.random.uniform(-1., 1., size=[batch, 128]), dtype='float32')


# 高斯分布噪声
def create_gaussian_noise(batch):
    return np.array(np.random.normal(loc=0.0, scale=1, size=(batch, 128)), dtype='float32')


# 随机均匀分布噪声
def create_noise_v2(m, n):
    return np.array(np.random.uniform(0., 1., size=[m, n]), dtype='float32')


# 随机获取生成器的训练集
def get_generator_train_batch():
    ones = np.ones(shape=(batch_size, 1), dtype=np.float)
    zeros = np.zeros(shape=(batch_size, 1), dtype=np.float)
    # 此时这里应构建真实样本的标签，即在训练生成器时，我们希望骗过判别器
    batch_real_labels = np.hstack((ones, zeros))
    train_x = create_gaussian_noise(batch_size)
    return train_x, batch_real_labels


# 随机获取判别器训练集(真实样本+生成器生成的样本)
def get_discriminator_train_batch(sess, graph_tensor_by_fake_images, noise_placeholder, real_images):
    mini_batch_size = batch_size // 2
    ones = np.ones(shape=(mini_batch_size, 1), dtype=np.float)
    zeros = np.zeros(shape=(mini_batch_size, 1), dtype=np.float)
    batch_real_labels = np.hstack((ones, zeros))
    batch_fake_labels = np.hstack((zeros, ones))
    # 真样本
    train_batch_images = data_batch_set(real_images)
    train_batch_images = train_batch_images[0:mini_batch_size]
    # 随机噪声
    train_batch_zs = create_gaussian_noise(mini_batch_size)

    # 生成器生成的假样本(使用generator_inference推理速度慢，具体原因不清楚)
    # train_batch_fake_images = sess.run(generator_inference(train_batch_zs, reuse_variables=True))
    train_batch_fake_images = sess.run(graph_tensor_by_fake_images, feed_dict={noise_placeholder: train_batch_zs})

    real_train_set = np.hstack((train_batch_images, batch_real_labels))
    fake_train_set = np.hstack((train_batch_fake_images, batch_fake_labels))
    train_set = np.vstack((real_train_set, fake_train_set))
    # 打乱顺序
    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(train_set)

    # 一个样本的长度
    one_len = len(train_batch_images[0])
    # 分离样本与标签
    train_images = train_set[:, 0:one_len]
    train_labels = train_set[:, one_len:one_len + 2]
    return train_images, train_labels


# 随机获取判别器训练集(真实样本+生成器生成的样本)
def get_discriminator_train_batch_v2(sess, graph_tensor_by_fake_images, noise_placeholder, real_images, real_set=None):
    mini_batch_size = batch_size // 2
    ones = np.ones(shape=(mini_batch_size, 1), dtype=np.float)
    zeros = np.zeros(shape=(mini_batch_size, 1), dtype=np.float)

    # 随机噪声
    train_batch_zs = create_gaussian_noise(mini_batch_size)
    # print(real_set)
    train_set = []
    if real_set is True:
        # 真样本
        train_batch_images = data_batch_set(real_images)
        train_batch_images = train_batch_images[0:mini_batch_size]
        batch_real_labels = np.hstack((ones, zeros))
        train_set = np.hstack((train_batch_images, batch_real_labels))

    elif real_set is False:
        # 生成器生成的假样本(使用generator_inference推理速度慢，具体原因不清楚)
        # train_batch_fake_images = sess.run(generator_inference(train_batch_zs, reuse_variables=True))
        train_batch_fake_images = sess.run(graph_tensor_by_fake_images, feed_dict={noise_placeholder: train_batch_zs})
        batch_fake_labels = np.hstack((zeros, ones))
        train_set = np.hstack((train_batch_fake_images, batch_fake_labels))

    elif real_set is None:

        # 真样本
        train_batch_images = data_batch_set(real_images)
        train_batch_images = train_batch_images[0:mini_batch_size]
        batch_real_labels = np.hstack((ones, zeros))
        real_train_set = np.hstack((train_batch_images, batch_real_labels))

        # 生成器生成的假样本(使用generator_inference推理速度慢，具体原因不清楚)
        # train_batch_fake_images = sess.run(generator_inference(train_batch_zs, reuse_variables=True))
        train_batch_fake_images = sess.run(graph_tensor_by_fake_images, feed_dict={noise_placeholder: train_batch_zs})
        batch_fake_labels = np.hstack((zeros, ones))
        fake_train_set = np.hstack((train_batch_fake_images, batch_fake_labels))
        train_set = np.vstack((real_train_set, fake_train_set))

    # 打乱顺序
    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(train_set)

    # 一个样本的长度
    one_len = len(train_set[0]) - 2
    # 分离样本与标签
    train_images = train_set[:, 0:one_len]
    train_labels = train_set[:, one_len:one_len + 2]
    return train_images, train_labels


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

    # 训练判别器的样本标签
    d_labels = tf.compat.v1.placeholder(tf.float32, [None, 2])
    # 训练生成器的样本标签
    g_labels = tf.compat.v1.placeholder(tf.float32, [None, 2])

    # 判别器输入
    d_x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    # 生成器输入
    g_x = tf.compat.v1.placeholder(tf.float32, [None, 128])
    # 判别器前向传播
    d_p = discriminator_inference(input_tensor=d_x)
    # 生成器前向传播
    fake_images = generator_inference(input_tensor=g_x)
    # 生成器串联判别器
    g_d_p = discriminator_inference(fake_images, reuse_variables=True)

    # 损失函数
    cross_entropy_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_p, labels=d_labels))
    cross_entropy_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=g_d_p, labels=g_labels))

    # 初始速率0.9，后面每训练100次后在学习速率基础上乘以0.96
    learning_rate_d = tf.compat.v1.train.exponential_decay(0.0001, global_step_d, 500, 0.9, staircase=True)
    learning_rate_g = tf.compat.v1.train.exponential_decay(0.0001, global_step_g, 500, 0.9, staircase=True)

    # 获取需要训练的变量
    g_vars, d_vars = get_trainable_variables()
    # 使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数。
    train_step_d = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(loss=cross_entropy_d
                                                                                              ,
                                                                                              global_step=global_step_d
                                                                                              , var_list=d_vars)
    train_step_g = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005).minimize(loss=cross_entropy_g
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
        d_loss = 0
        g_loss = 0

        for i in range(TRAINING_STEPS):
            # 训练D
            for k in range(20):
                # 获取训练数据，包含了真样本与生成器生成的样本
                train_images, train_labels = get_discriminator_train_batch_v2(sess=sess
                                                                              , graph_tensor_by_fake_images=fake_images
                                                                              , noise_placeholder=g_x
                                                                              , real_images=real_images
                                                                              , real_set=None)
                _, d_loss = sess.run([train_step_d, cross_entropy_d],
                                     feed_dict={d_x: train_images, d_labels: train_labels})
                if k % 5 == 0:
                    # loss
                    print("Discriminator training, d_loss=%f" % (d_loss))

            # 训练G
            fake_image_nps=[]
            for k in range(20):
                # 获取训练数据
                train_z, train_labels = get_generator_train_batch()
                _, g_loss, fake_image_nps = sess.run([train_step_g, cross_entropy_g, fake_images],
                                                     feed_dict={g_x: train_z, g_labels: train_labels})
                if k % 5 == 0:
                    # loss
                    print("Generator training, g_loss=%f" % (g_loss))

            if i % 2 == 0:
                save_images(sess, fake_image_nps)


def main():
    train()


if __name__ == '__main__':
    main()

```

