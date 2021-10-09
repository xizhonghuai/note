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



### fashion_mnist数据重建实例

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
