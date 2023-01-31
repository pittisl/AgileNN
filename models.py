import tensorflow as tf
import numpy as np


class Quantization_Layer(tf.keras.layers.Layer):
    """
    Adapted from
    Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations
    https://arxiv.org/abs/1704.00648
    """
    def __init__(self, num_centroids=8, name='my_quantization_layer'):
        super(Quantization_Layer, self).__init__()
        self.centroids = self.add_weight(shape=(1, num_centroids), initializer="random_normal", trainable=True, name='q_centroids')

    def call(self, x):
        centroids = 6.0 * tf.nn.sigmoid(self.centroids) # fall in relu6 range (0, 6). It's the key to stable training
        c_indices = tf.math.argmin(
            tf.math.square(x[:, :, :, :, tf.newaxis] - centroids[tf.newaxis, tf.newaxis, tf.newaxis, :, :]),
            axis=-1) # (None, 32, 32, 19)
            
        y_hard_1 = tf.gather(centroids, c_indices, axis=-1, batch_dims=0) # (1, None, 32, 32, 19)
        y_hard = tf.squeeze(y_hard_1, axis=0)

        y_soft_1 = tf.nn.softmax(
            -tf.math.square(x[:, :, :, :, tf.newaxis] - centroids[tf.newaxis, tf.newaxis, tf.newaxis, :, :]), 
            axis=-1) * centroids[tf.newaxis, tf.newaxis, tf.newaxis, :, :] # (None, 32, 32, 19, c)
        y_soft = tf.math.reduce_sum(y_soft_1, axis=-1)

        return y_soft + tf.stop_gradient(y_hard - y_soft), c_indices


def _conv_bn_lite(out, strides=3, dilation_rate=3, data_format='channels_last'):
    axis = 1 if data_format == "channels_first" else 3
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(4, 3, strides=strides, dilation_rate=dilation_rate, padding='same', use_bias=False, data_format=data_format),
        tf.keras.layers.BatchNormalization(axis=axis),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.SeparableConv2D(out, 3, strides=1, padding='same', use_bias=False, data_format=data_format),
        tf.keras.layers.BatchNormalization(axis=axis),
        tf.keras.layers.ReLU(max_value=6),
    ], name="fe")


def _local_predictor(num_classes, data_format):
    return tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(data_format=data_format),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.BatchNormalization(),
    ], name="l_predictor")


def _remote_predictor(num_classes, nn_config, input, data_format):
    input_channel = input
    nn_stacks = []
    for i, (t, c, n, s) in enumerate(nn_config):
        output_channel = c
        layers = []
        for j in range(n):
            if j == 0:
                layers.append(_InvertedResidual(i, j, input_channel, output_channel, s, expand_ratio=t, data_format=data_format))
            else:
                layers.append(_InvertedResidual(i, j, input_channel, output_channel, 1, expand_ratio=t, data_format=data_format))
            input_channel = output_channel
        nn_stacks.append(tf.keras.Sequential(layers, name=f'r_block{i}'))
    entire_body = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(16, 3, strides=1, padding='same', use_bias=False, data_format=data_format),
        tf.keras.Sequential(nn_stacks, name="r_main_body"),
        _conv1x1_bn(1280, data_format=data_format),
        tf.keras.layers.GlobalAveragePooling2D(data_format=data_format),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.BatchNormalization(),
    ], name="r_predictor")
    return entire_body


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=2, name='reweighting_layer'):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(1, units), initializer="random_normal", trainable=True, name='reweighting_w'
        )

    def call(self, local_out, remote_out, T=1.0):
        w_normalized = tf.nn.softmax(self.w / T, axis=-1)
        w1 = w_normalized[:, 0] # (1, 1)
        w2 = w_normalized[:, 1] # (1, 1)
        return tf.math.multiply(local_out, w1) + tf.math.multiply(remote_out, w2)


def _conv1x1_bn(out, data_format):
    axis = 1 if data_format == "channels_first" else 3
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out, 1, use_bias=False, data_format=data_format),
        tf.keras.layers.BatchNormalization(axis=axis),
        tf.keras.layers.ReLU(max_value=6),
    ])


class _InvertedResidual(tf.keras.Model):
    def __init__(self, i, j, inp, out, strides, expand_ratio, data_format=None):
        super(_InvertedResidual, self).__init__()
        assert strides in [1, 2]
        self._strides = strides
        hidden_dim = round(inp * expand_ratio)
        self._out = out
        self._use_res_connect = strides == 1 and inp == out
        self._data_format = data_format or 'channels_last'
        assert data_format in ['channels_first', 'channels_last']
        axis = 1 if data_format == "channels_first" else 3

        if expand_ratio == 1:
            self.conv = tf.keras.Sequential([
                # depth-wise
                tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # point-wise
                tf.keras.layers.Conv2D(out, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
            ], name=f'sepconv{i}_{j}')
        else:
            self.conv = tf.keras.Sequential([
                # point-wise
                tf.keras.layers.Conv2D(hidden_dim, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # depth-wise
                tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # point-wise
                tf.keras.layers.Conv2D(out, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
            ], name=f'sepconv{i}_{j}')

    def call(self, inputs, training=True):
        if self._use_res_connect:
            return inputs + self.conv(inputs, training=training)
        return self.conv(inputs, training=training)

    def compute_output_shape(self, input_shape):
        if self._data_format == 'channels_last':
            batch_size, height, width, channels = input_shape
        else:
            batch_size, channels, height, width = input_shape
        if self._strides == 2:
            height = height // tf.Dimension(2)
            width = width // tf.Dimension(2)
        if self._data_format == 'channels_last':
            return tf.TensorShape([batch_size, height, width, self._out])
        else:
            return tf.TensorShape([batch_size, self._out, height, width])


class MobileNetV2_AgileNN(tf.keras.Model):
    def __init__(self, classes=100, data_format=None, conv1_stride=3, split_ratio=0.2, num_centroids=8):
        super(MobileNetV2_AgileNN, self).__init__()
        data_format = data_format or 'channels_last'
        assert data_format in ['channels_first', 'channels_last']
        self.K_top = int(np.round(split_ratio * 24))
        width_multiplier = 1.5
        input_channel = int(width_multiplier * 16) # 24
        inverted_residual_config = [
            # t (expand ratio), channel, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.feature_extractor_1 = _conv_bn_lite(input_channel, conv1_stride, 1, data_format=data_format) # stride=2 for tinyimagenet   
        self.local_predictor_1 = _local_predictor(classes, data_format=data_format)
        self.remote_predictor_1 = _remote_predictor(classes, inverted_residual_config, 16, data_format=data_format)
        self.q_layer = Quantization_Layer(num_centroids=num_centroids)
        self.reweighting_1 = Linear()
        
    def call(self, inputs, training=False):
        features = self.feature_extractor_1(inputs, training=training)
        
        top_features, bottom_features = self.feature_splitter(features)
        q_bottom_features, _ = self.q_layer(bottom_features, training=training)
        local_logits = self.local_predictor_1(top_features, training=training)
        remote_logits = self.remote_predictor_1(q_bottom_features, training=training)
        
        reweighted_final_outs = local_logits + remote_logits # self.reweighting_1(local_logits, remote_logits, training=training)
        return local_logits, remote_logits, reweighted_final_outs

    def feature_splitter(self, features): # (None, 32, 32, 24) split_ratio = 20%
        top_features = features[:, :, :, :self.K_top]
        bottom_features = features[:, :, :, self.K_top:]
        return top_features, bottom_features