# Necessary packages
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import sys
import numpy as np
from tqdm import tqdm
import os
import random
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import Model

def ek_cph(data_x,parameters,data_image,prescription_z,prescription_image):
    seed = 25
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = parameters['batch_size']
    hint_rate = parameters['hint_rate']
    alpha = parameters['alpha']
    # beta = parameters['beta']
    iterations = parameters['iterations']

    # Other parameters
    no, dim = data_x.shape
    h_dim = int(dim)  # Hidden state dimensions

    no, dim_z = prescription_z.shape
    # h_dim_z = int(dim_z)# Hidden state dimensions
    # print(h_dim)

    # Normalization，Normalize data in [0, 1] range.
    norm_data, norm_parameters_x = normalization(data_x)
    # norm_data_x = np.nan_to_num(norm_data, 0)
    norm_data_x = np.nan_to_num(data_x, 0)
    norm_data_z = np.nan_to_num(prescription_z, 0)

    # Input placeholders
    X_pre = tf.placeholder(tf.float32, shape=[1, 483, dim, 2])
    Z_pre = tf.placeholder(tf.float32, shape=[1, 483, dim_z, 2], name='Z_pre_placeholder')

    # Data vector
    # X = tf.placeholder(tf.float32, shape = [None, dim])
    M = tf.placeholder(tf.float32, shape=[None, dim])# Mask vector
    H = tf.placeholder(tf.float32, shape=[None, dim])# Hint vector

    # Discriminator variables 发病率参数
    D_W1_x = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1_x = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2_x = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2_x = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3_x = tf.Variable(xavier_init([h_dim, dim]))
    D_b3_x = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs
    theta_D_x = [D_W1_x, D_W2_x, D_W3_x, D_b1_x, D_b2_x, D_b3_x]

    # Generator variables
    conv_filter_w1_x = tf.Variable(tf.random_normal([1, 4, 2, 2]))
    conv_filter_b1_x = tf.Variable(tf.random_normal([2]))
    conv_filter_w2_x = tf.Variable(tf.random_normal([1, 4, 2, 1]))
    conv_filter_b2_x = tf.Variable(tf.random_normal([1]))
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1_x = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1_x = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2_x = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2_x = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3_x = tf.Variable(xavier_init([h_dim+dim, dim]))
    G_b3_x = tf.Variable(tf.zeros(shape=[dim]))
    theta_G_x = [G_W1_x, G_W2_x, G_W3_x, G_b1_x, G_b2_x, G_b3_x, conv_filter_w1_x, conv_filter_b1_x, conv_filter_w2_x,
               conv_filter_b2_x]  #

    ## CPH functions
    # CNN + Generator

    def cnn(x):
        relu_feature_maps1 = tf.nn.tanh( \
            tf.nn.conv2d(x, conv_filter_w1_x, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1_x)
        max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1, 1, 4, 1], strides=[1, 1, 1, 1], padding='SAME')  #kize[batch, height, width, channels]

        relu_feature_maps2 = tf.nn.tanh( \
            tf.nn.conv2d(max_pool1, conv_filter_w2_x, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b2_x)
        max_pool2 = tf.nn.max_pool(relu_feature_maps2, ksize=[1, 1, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

        x0 = tf.reshape(max_pool2, [483, dim])
        return x0

    def generator(x1, m, x2):
        x1 = cnn(x1)
        # print('x1:',x1.shape)
        x2 = cnn(x2)
        # print('x2:',x2.shape)
        # x3 = x1 + x2
        # x1 = cbam_block(x1,ratio=8)
        # x2 = cbam_block(x2,ratio=8)
        # print(x1,x2)
        inputs = tf.concat(values=[x1, m], axis=1)
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1_x) + G_b1_x)
        # G_h1 = tf.concat(values=[G_h1,x2],axis=1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2_x) + G_b2_x)
        G_h2 = tf.concat(values=[G_h2,x2],axis=1)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3_x) + G_b3_x)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.tanh(tf.matmul(inputs, D_W1_x) + D_b1_x)
        D_h2 = tf.nn.tanh(tf.matmul(D_h1, D_W2_x) + D_b2_x)
        D_logit = tf.matmul(D_h2, D_W3_x) + D_b3_x
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## CPH structure
    # Generator
    G_sample = generator(X_pre, M, Z_pre)
    X2 = X_pre[0, :, :, 0]
    Z2 = Z_pre[0, :, :, 0]
    # Combine with observed data
    Hat_X = X2 * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## CPH loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(
        1. - D_prob + 1e-8))  # tf.reduce_mean

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X2 - M * G_sample) ** 2) / tf.reduce_mean(M)

    Huber_loss = tf.reduce_mean(tf.losses.huber_loss(Z2,G_sample,weights=1.0,delta=1.0))

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss + Huber_loss
    # G_loss = G_loss_temp + alpha * MSE_loss
    ## CPH solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss,
                                                 var_list=theta_D_x)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G_x)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        # print('baych_idx:',len(batch_idx))#baych_idx: 483
        image_mb = data_image[:, batch_idx, :, :]
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)

        H_mb = M_mb * H_mb_temp
        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
        image_mb[0, :, :, 0] = X_mb
        # print('M_mb:',M_mb.shape)#M_mb: (483, 8)
        # print('X_pre',X_pre.shape)#X_pre (1, 483, 8, 2)
        # print('H',H_mb.shape)#H (483, 8)

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X_pre: image_mb, H: H_mb, Z_pre: prescription_image})
        _, G_loss_curr, MSE_loss_curr, Huber_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss, Huber_loss],
                     feed_dict={X_pre: image_mb, M: M_mb, H: H_mb, Z_pre: prescription_image})
        # _, G_loss_curr, MSE_loss_curr, = \
        #     sess.run([G_solver, G_loss_temp, MSE_loss,],
        #              feed_dict={X_pre: image_mb, M: M_mb, H: H_mb, Z_pre: prescription_image})


    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    image_mb = data_image
    image_mb[0, :, :, 0] = X_mb

    # image_mb_z = prescription_image
    # image_mb_z[0, : ,: ,0] = norm_data_z
    # print('image_mb',image_mb.shape)
    # print('M_mb',M_mb.shape)
    # print('image_mb',image_mb_z)
    imputed_data = sess.run([G_sample], feed_dict={X_pre: image_mb, M: M_mb, Z_pre: prescription_image})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    # imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data





