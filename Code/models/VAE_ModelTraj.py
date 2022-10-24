import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU

from Code.training.schedulers import CustomSchedule, HalveSchedule
from Code.utils.save_utils import load_pkl_data, save_pkl_data, valid_file
from Code.eval.quantitative_eval import ADE

# utilities
import os
import random
from glob import glob
import pathlib
import time
import datetime

gpu_available = tf.config.list_physical_devices('GPU')
print(gpu_available)


def get_look_ahead_mask(input_data):
    input_shape = list(input_data.shape)[:-1]
    input_shape.insert(-1, input_shape[-1])
    input_shape.insert(1, 1)
    mask = 1 - tf.linalg.band_part(tf.ones(input_shape), -1, 0)
    return mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_position, d_model):
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def mask_output(output, masks, mode='seq'):
    if mode == 'seq':
        mod_masks = (1 - masks)[:, :, :,
                    tf.newaxis]  # (batch, <copy to neighbors dim>, seq, <copy_mask to match feats dim>)
    else:
        mod_masks = (1 - masks)[:, :, :, tf.newaxis]  # (batch, seq, neighbors, <copy to feat dim>)
    return output * mod_masks


def ScaledDotProduct(Q, K, V, mask=None):
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # compute attention
    KT = tf.transpose(K, [0, 1, 2, 4, 3])
    attention = tf.matmul(Q, KT) / tf.sqrt(dk)

    # mask if necessary
    if mask is not None:
        # print(attention.shape)
        attention += (mask * -1e9)

    # compute values and weighted sum of their attention
    # weights = tf.nn.softmax(attention, axis=-1)
    weights = tf.nn.sigmoid(attention)
    # weights = tf.nn.sigmoid(attention)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, dk=256, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        # params
        self.num_heads = num_heads
        self.dk = dk
        self.dk_by_head = dk // num_heads

        # layers
        self.WQ = keras.layers.Dense(dk)
        self.WK = keras.layers.Dense(dk)
        self.WV = keras.layers.Dense(dk)
        self.dense = keras.layers.Dense(dk)

    def splitheads(self, x):
        batch_size, seq_length = x.shape[0:2]
        # spliting the heads done by reshaping last dimension
        x = tf.reshape(x, (
        batch_size, seq_length, -1, self.num_heads, self.dk_by_head))  # (batch, seq, neighbors, head, features_by_head)
        return tf.transpose(x, (0, 3, 1, 2, 4))  # (batch, head, seq, neighbors, features_by_head)

    def call(self, q, k, v, mask=None):
        batch_size, seq_length = q.shape[0:2]

        # projections
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        # split heads
        q = self.splitheads(q)
        k = self.splitheads(k)
        v = self.splitheads(v)

        # compute attention and merge heads
        attn_output, attention = ScaledDotProduct(q, k, v, mask)  # (batch, head, seq, neighbors, features_by_head)
        attn_output = tf.transpose(attn_output, (0, 2, 3, 1, 4))  # (batch, seq, neighbors, head, features_by_head)
        concat_output = tf.reshape(attn_output,
                                   (batch_size, seq_length, -1, self.dk))  # (batch, seq, neighbors, features)
        output = self.dense(concat_output)

        return output, attention


def get_ffn(d_model, hidden_size, act_func='relu'):
    return keras.models.Sequential([
        keras.layers.Dense(hidden_size, activation=act_func),
        keras.layers.Dense(d_model)
    ])


class EncoderLayer(keras.layers.Layer):
    def __init__(self, dk=256, num_heads=8, hidden_layer_size=256, drop_rate=0.1):
        super(EncoderLayer, self).__init__()

        # layers
        self.MH = MultiHeadAttention(dk, num_heads)
        #self.ffn = get_ffn(dk, hidden_layer_size)
        self.normLayer1 = keras.layers.LayerNormalization(epsilon=1e-6)
        #self.normLayer2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(drop_rate)
        #self.dropout2 = keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        if type(x) in (list, tuple):
            k = x[1]
            x = x[0]
        else:
            k = x

        # multihead attention
        attn_output, weights = self.MH(x, k, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        z = self.normLayer1(x + attn_output)
        # normalization and feed forward layers
        #output = self.ffn(z)
        #output = self.dropout2(output, training=training)
        #output = self.normLayer2(z + output)

        return z, weights


class DecoderLayer(keras.layers.Layer):
    def __init__(self, dk=256, num_heads=8, hidden_layer=256, drop_rate=0.1):
        super(DecoderLayer, self).__init__()
        # layers
        self.SAMH = MultiHeadAttention(dk, num_heads)
        self.EDMH = MultiHeadAttention(dk, num_heads)
        #self.ffn = get_ffn(dk, hidden_layer)

        self.normLayer1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.normLayer2 = keras.layers.LayerNormalization(epsilon=1e-6)
        #self.normLayer3 = keras.layers.LayerNormalization(epsilon=1e-6)\

        self.dropout1 = keras.layers.Dropout(drop_rate)
        self.dropout2 = keras.layers.Dropout(drop_rate)
        #self.dropout3 = keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # self attention computation
        self_attn_out, self_attn = self.SAMH(x, x, x, look_ahead_mask)
        self_attn_out = self.dropout1(self_attn_out, training=training)
        z = self.normLayer1(x + self_attn_out)

        # encoder decoder computation
        enc_dec_out, enc_dec_attn = self.EDMH(z, enc_output, enc_output, padding_mask)
        enc_dec_out = self.dropout2(enc_dec_out, training=training)
        z = self.normLayer2(z + enc_dec_out)

        # feed forward computation
        #output = self.ffn(z)
        #output = self.dropout3(output, training=training)
        #output = self.normLayer3(z + output)

        return z, self_attn, enc_dec_attn


class Encoder(keras.layers.Layer):
    def __init__(self, features_size, max_size, dk_model=256, num_heads=8, num_encoders=6,
                 enc_hidden_size=256, use_pos_emb=True, drop_rate=0.1):
        super(Encoder, self).__init__()

        # params
        self.dk_model = dk_model
        self.max_size = max_size
        self.use_pos_emb = use_pos_emb
        self.enc_hidden_size = enc_hidden_size
        self.num_encoders = num_encoders

        # layers
        self.positional_encoding = positional_encoding(self.max_size, self.dk_model)
        self.embedding = keras.layers.Dense(dk_model)
        self.encoders_stack = [EncoderLayer(dk_model, num_heads, enc_hidden_size, drop_rate) for _ in
                               range(num_encoders)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, padding_mask, training):
        k = None
        if type(x) in (list, tuple):
            k = x[1]
            x = x[0]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dk_model, tf.float32))
        if self.use_pos_emb:
            x += self.positional_encoding

        x = self.dropout(x, training=training)

        for encoder_layer in self.encoders_stack:
            args = [x, k] if k is not None else x
            x, weights = encoder_layer(args, training, padding_mask)

        return x, weights


class Decoder(keras.layers.Layer):
    def __init__(self, features_size, max_size, dk_model=256, num_heads=8, num_decoders=6,
                 dec_hidden_size=256, use_pos_emb=True, drop_rate=0.1):

        super(Decoder, self).__init__()

        # params
        self.dk_model = dk_model
        self.max_size = max_size
        self.use_pos_emb = use_pos_emb
        self.dec_hidden_size = dec_hidden_size
        self.num_decoders = num_decoders
        self.positional_encoding = positional_encoding(self.max_size, self.dk_model)

        # layers
        self.embedding = keras.layers.Dense(dk_model)
        self.decoders_stack = [DecoderLayer(dk_model, num_heads, dec_hidden_size, drop_rate) for _ in
                               range(num_decoders)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dk_model, tf.float32))
        if self.use_pos_emb:
            x += self.positional_encoding

        x = self.dropout(x, training=training)
        for decoder_layer in self.decoders_stack:
            x, attn1, attn2, = decoder_layer(x, enc_output, training, look_ahead_mask, padding_mask)

        return x


class Sampler(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mean_layer = keras.layers.Dense(self.latent_dim, name="z_mean")
        self.var_layer = keras.layers.Dense(self.latent_dim, name="z_log_var")

    def get_config(self):
        config = super(Sampler, self).get_config()
        config.update({"units": self.units})
        return config

    def call(self, input_data):
        '''
        input_dim is a vector in the latent (codified) space
        '''
        z_mean = self.mean_layer(input_data)
        z_log_var = self.var_layer(input_data)

        shape = z_mean.shape

        epsilon = tf.keras.backend.random_normal(shape=shape)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z, z_mean, z_log_var


class SemanticMapFeatures(keras.layers.Layer):
    def __init__(self, N, neighbors, out_dims, kernel_sizes, strides):
        super(SemanticMapFeatures, self).__init__()
        self.N = N
        self.neighbors = neighbors
        # self.ConvLayers = [keras.layers.Conv2D(out_dims[i], kernel_sizes[i], strides=strides[i], data_format='channels_first') for i in range(N)]
        # self.reshape = keras.layers.Reshape([-1, neighbors, 28 * 28])
        self.ConvLayers = []
        self.dense = tf.keras.layers.Dense(32, activation='relu')
        h, w, c = 256, 256, 3
        for i in range(N):
            self.ConvLayers.append(
                keras.layers.Conv2D(out_dims[i], kernel_sizes[i], strides=strides[i]))
            h = (h - kernel_sizes[i]) // 2 + 1
            w = (w - kernel_sizes[i]) // 2 + 1
            c = out_dims[i]

    def call(self, inputs, neighs):
        output = inputs
        output = tf.reshape(output, [-1, 256, 256, 3])
        for layer in self.ConvLayers:
            output = layer(output)

        output = tf.keras.activations.tanh(output)
        output = tf.reshape(output, [-1, neighs, 12 * 12])
        output = self.dense(output)
        return output


class STTransformer(keras.Model):
    def __init__(self, features_size, seq_size, neigh_size,
                 sp_dk=256, sp_enc_heads=8, sp_dec_heads=8, sp_num_encoders=6, sp_num_decoders=6,
                 tm_dk=256, tm_enc_heads=8, tm_dec_heads=8, tm_num_encoders=6, tm_num_decoders=6,
                 batch=1):
        super(STTransformer, self).__init__()

        self.seq_size = seq_size
        self.neigh_size = neigh_size
        self.batch_size = batch
        # layers
        #self.feat_embedding = keras.layers.Dense(144)
        #self.semantic_map = SemanticMapFeatures(4, neigh_size, out_dims=[16, 16, 16, 1], kernel_sizes=[5, 5, 5, 7], strides=[2, 2, 2, 2])
        #self.sampler = Sampler(tm_dk)

        # spatial
        self.sp_encoder = Encoder(features_size, neigh_size, sp_dk, num_heads=sp_enc_heads, num_encoders=sp_num_encoders, use_pos_emb=False)

        # time
        self.tm_encoder = Encoder(features_size, seq_size, tm_dk, num_heads=tm_enc_heads, num_encoders=tm_num_encoders, use_pos_emb=True)
        self.tm_decoder = Decoder(features_size, seq_size, tm_dk, num_heads=tm_dec_heads, num_decoders=tm_num_decoders, use_pos_emb=True)

        # self.decoder = Decoder(features_size, neigh_size, sp_dk, num_heads=sp_dec_heads, num_decoders=sp_num_decoders, use_pos_emb=False)
        self.linear = tf.keras.layers.Dense(3)
        # training
        self.loss_object = tf.keras.losses.MeanSquaredError(reduction='sum')
        self.ownloss_weights = tf.constant([(1 + 0.01) ** i for i in range(self.seq_size + 1)])[tf.newaxis, :, tf.newaxis, tf.newaxis]
        self.optimizer = None
        # eval loss different because batch is not divided in two GPUs
        self.eval_loss = tf.keras.losses.MeanSquaredError()

    @tf.function
    def encode(self, inputs, training):
        past, past_speed, past_seq_masks, past_neigh_masks, past_speed_masks, extra_neigh_masks, maps = inputs
        _, _, neighs, _ = past.shape

        past = self.feat_embedding(past)
        proc_maps = self.semantic_map(maps, neighs)
        # multiply by ones to match all neighbors shape, except features dim
        sp_desired_shape = past.shape[:-1] + proc_maps.shape[-1]
        sp_proc_maps = proc_maps[:, tf.newaxis, :, :] * tf.ones(sp_desired_shape)
        # concat features embeddings and feature maps
        past = tf.concat((past, sp_proc_maps), axis=-1)
        # spatial transformer
        sp_out, weights = self.sp_encoder(past, past_neigh_masks, training)
        sp_out = sp_out[:, 1:, :, :] - sp_out[:, :-1, :, :]
        output = tf.transpose(sp_out, [0, 2, 1, 3])
        # time transformer
        time_input = [past_speed, output]
        output = self.tm_encoder(time_input, past_speed_masks, training)
        return output, sp_out, weights
    
    # to switch between wocnn or just the transformer, uncomment the above lines and change past_speed by output in time_input = [...]
    @tf.function
    def encode_abl(self, inputs, training):
        past, past_speed, past_seq_masks, past_neigh_masks, past_speed_masks, extra_neigh_masks, _ = inputs
        _, _, neighs, _ = past.shape

        # toggle comments from here
        sp_out, weights = self.sp_encoder(past, past_neigh_masks, training)
        sp_out = sp_out[:, 1:, :, :] - sp_out[:, :-1, :, :]
        output = tf.transpose(sp_out, [0, 2, 1, 3])
        
        # time transformer
        time_input = [past_speed, output]
        #time_input = [past_speed, past_speed]
        output = self.tm_encoder(time_input, past_speed_masks, training)
        #return output, None
        return output, sp_out, weights

    @tf.function
    def decode(self, inputs, training):
        future, targets, tar_masks, inp_masks, enc_out, sp_enc_out = inputs

        look_mask = get_look_ahead_mask(targets)
        look_mask = tf.maximum(look_mask, tar_masks)
        output = self.tm_decoder(targets, enc_out, look_mask, inp_masks, training)

        output = tf.transpose(output, [0, 2, 1, 3])  # (batch, seq, neigh, [x,y])

        # EXPERIMENTAL
        #output = tf.concat([output, sp_enc_out], axis=-1)
        # output = tf.concat([output, sp_out], axis=-1)
        output = self.linear(output)
        # END EXPERIMENTAL

        output = tf.concat([future[:, 0:1, :, :], output], axis=1)
        # output = mask_output(output, squeezed_neigh_mask, 'neigh')
        output = tf.math.cumsum(output, axis=1)
        # output = self.decoder(output, sp_out, None, futu_neigh_masks, training)
        # output = self.linear(output)
        return output

    @tf.function
    def call(self, inputs, training, stds):
        """
          speeds.shape = (batch, neighbors, seq, feats)
        """
        past_speed_masks, enc_out, sp_enc_out = inputs[0]
        future, future_speed, _, futu_neigh_masks, futu_speed_masks = inputs[1]

        output = self.decode([future, future_speed, futu_speed_masks, past_speed_masks, enc_out, sp_enc_out], training)
        return output

    def loss_function(self, real, pred):
        # loss_ = (tf.reduce_sum(((real-pred)**2)*self.ownloss_weights)/3) * (1. / (self.seq_size * self.neigh_size * self.batch_size))
        loss_ = self.loss_object(real, pred) * (1. / (self.seq_size * self.neigh_size * self.batch_size))
        return loss_

    @tf.function
    def iterative_train_step(self, inputs):
        past, future, maps, stds = inputs
        neigh_out_masks = tf.squeeze(future[3])

        with tf.GradientTape() as tape:
            predictions, weights = self.inference((past, future, maps), stds, True)
            masked_predictions = mask_output(predictions, neigh_out_masks, 'neigh')
            loss = self.loss_function(future[0], masked_predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 2.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def eval_step(self, past, future, maps):
        squeezed_mask = tf.squeeze(future[3])
        preds, weights = self.inference((past, future, maps), None, False)
        preds = mask_output(preds, squeezed_mask, 'neigh')
        eval_loss = self.loss_function(future[0], preds)
        return preds, eval_loss, weights

    @tf.function
    def inference(self, inputs, stds, training):
        past = inputs[0]
        maps = inputs[2]
        past_speed_masks = inputs[0][4]

        target, _, _, neigh_masks, speed_masks = inputs[1]

        tar_sequence = tf.TensorArray(dtype=tf.float32, size=self.seq_size + 1)
        tar_sequence = tar_sequence.write(0, target[:, 0, :, :])
        tar_sequence = tar_sequence.write(1, target[:, 1, :, :])

        enc_out, sp_enc_out, weights = self.encode_abl([*past, maps], training)
        past_info = [past_speed_masks, enc_out, sp_enc_out]

        for i in range(2, self.seq_size + 1):
            future_seq = tf.transpose(tar_sequence.stack(), [1, 0, 2, 3])  # transpose to get batch dimension first
            future_speed = (future_seq[:, 1:, :, :] - future_seq[:, :-1, :, :])
            future_speed = tf.transpose(future_speed, [0, 2, 1, 3])
            future = [future_seq, future_speed, None, neigh_masks, speed_masks]
            # predict
            output = self((past_info, future), training, stds)
            # append new prediction
            tar_sequence = tar_sequence.write(i, output[:, i, :, :])

        return tf.transpose(tar_sequence.stack(), [1, 0, 2, 3]), weights

    @staticmethod
    def get_model_params(params):
        if params.get('features_size') is None or params.get('seq_size') is None or \
                params.get('neigh_size') is None or params.get('batch') is None:
            raise RuntimeError(
                '[ERR] parameters file should contain basic model params (feat_size, seq_size, neigh_size, batch)')
        model_params = {
            'features_size': params['features_size'],
            'seq_size': params['seq_size'],
            'neigh_size': params['neigh_size'],
            'sp_dk': params.get('sp_dk', 256),
            'sp_enc_heads': params.get('sp_enc_heads', 4),
            'sp_dec_heads': params.get('sp_dec_heads', 4),
            'tm_dk': params.get('tm_dk', 256),
            'tm_enc_heads': params.get('sp_enc_heads', 4),
            'tm_dec_heads': params.get('sp_dec_heads', 4),
            'sp_num_encoders': params.get('sp_num_encoders', 4),
            'sp_num_decoders': params.get('sp_num_decoders', 4),
            'tm_num_encoders': params.get('tm_num_encoders', 4),
            'tm_num_decoders': params.get('tm_num_decoders', 4),
            'batch': params['batch']
        }
        return model_params

    def get_optimizer(self, dk, preload, save_path, config_path=None, params=None):
        if params is None:
            params = {}

        # load lr parameter from parameters file, if no value found use fixed lr
        lr = params.get('lr', 0.00001)

        # if lr value found is int, use that value as the warm up steps.
        if type(lr) is int:
            lr = CustomSchedule(dk, lr)

        #  if lr value is None, use default warmup steps
        elif lr is None:
            print('**************** [WARN]: using default value as warm up steps not desirable ************* ')
            lr = CustomSchedule(dk)

        # preload optimizer
        if config_path is not None and preload:
            # validate files
            valid_file(config_path)
            conf = load_pkl_data(config_path)
            if type(lr) is float:
                # note that if lr is a valid float, it will overwrite the 'learning_rate' obtained from  conf file
                conf['learning_rate'] = lr
            else:
                # use CustomSchedule loaded from the conf file
                conf['learning_rate'] = CustomSchedule.from_config(conf['learning_rate']['config'])

            # set optimizer
            self.optimizer = tf.keras.optimizers.Adam.from_config(conf)
        else:
            b1 = params.get('beta_1', 0.99)
            b2 = params.get('beta_2', 0.9)
            epsilon = params.get('epsilon', 1e-9)
            self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=b1, beta_2=b2, epsilon=epsilon)
        
        save_pkl_data(self.optimizer.get_config(), save_path, 4)

# model in which inputs of neihgbors and sequences will be in the same dimension

