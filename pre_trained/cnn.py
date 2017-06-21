import _pickle as cPickle
import numpy as np
import tensorflow as tf

class PretrainedCNN:
    def __init__(self, weight_path, mdl_name='vgg16'):
        self.mdl_name = mdl_name
        self.model = self.build_model(weight_path)
    def build_model(self, weight_path):
        if self.mdl_name == 'vgg16':
            self.params = []
            ws = cPickle.load(open(weight_path, 'rb'))
            self.img = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.label = tf.placeholder(tf.float32, [None, 1000])
            x = self.img
            x = self.conv('conv1_1', x, ws[:2])
            x = self.conv('conv1_2', x, ws[2:4])
            x = self.pool('pool1', x)
            x = self.conv('conv2_1', x, ws[4:6])
            x = self.conv('conv2_2', x, ws[6:8])
            x = self.pool('pool2', x)
            x = self.conv('conv3_1', x, ws[8:10])
            x = self.conv('conv3_2', x, ws[10:12])
            x = self.conv('conv3_3', x, ws[12:14])
            x = self.pool('pool3', x)
            x = self.conv('conv4_1', x, ws[14:16])
            x = self.conv('conv4_2', x, ws[16:18])
            x = self.conv('conv4_3', x, ws[18:20])
            x = self.pool('pool4', x)
            x = self.conv('conv5_1', x, ws[20:22])
            x = self.conv('conv5_2', x, ws[22:24])
            x = self.conv('conv5_3', x, ws[24:26])
            x = self.pool('pool5', x)
            s = int(np.prod(x.get_shape()[1:]))
            self.ext = tf.reshape(x, [-1, s])
            x = self.fc('fc1', self.ext, ws[26:28])
            x = self.fc('fc2', x, ws[28:30])
            self.out = self.fc('pred', x, ws[30:32], act=False)
    def conv(self, name, x, ws):
        with tf.name_scope(name) as scope:
            W = tf.Variable(ws[0], trainable=True, name='weights')
            b = tf.Variable(ws[1], trainable=True, name='biases')
            conv = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
            out = tf.nn.relu(tf.nn.bias_add(conv, b), name=scope)
            self.params += [W, b]
            return out
    def pool(self, name, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    def fc(self, name, x, ws, act=True):
        with tf.name_scope(name) as scope:
            W = tf.Variable(ws[0], trainable=True, name='weights')
            b = tf.Variable(ws[1], trainable=True, name='biases')
            out = tf.nn.bias_add(tf.matmul(x, W), b)
            out = tf.nn.relu(out) if act else out
            self.params += [W, b]
            return out
    def get_output(self, sess, X):
        layer_outputs = sess.run([self.ext], feed_dict={self.img:X})
        return layer_outputs