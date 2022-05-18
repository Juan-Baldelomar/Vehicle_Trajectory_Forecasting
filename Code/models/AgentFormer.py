
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU

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


def get_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones([size, size]), -1, 0)
    return mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
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
        mod_masks = (1-masks)[:, :, tf.newaxis, tf.newaxis]  # (batch, <copy to neighbors dim>, seq, <copy_mask to match feats dim>)
    else:
        mod_masks = (1-masks)[:, :, :, tf.newaxis]           # (batch, seq, neighbors, <copy to feat dim>)
    return output * mod_masks


def ScaledDotProduct(Q, K, V, mask=None):
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # compute attention
    KT = tf.transpose(K, [0, 1, 3, 2])
    attention = tf.matmul(Q, KT)/tf.sqrt(dk)

    # mask if necessary
    if mask is not None:
        #print(attention.shape)
        attention += (mask * -1e9)

    # compute values and weighted sum of their attention
    weights = tf.nn.softmax(attention, axis=-1)
    #weights = tf.nn.sigmoid(attention)
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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.dk_by_head))  # (batch, seq, head, features_by_head)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq, features_by_head)

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
        attn_output, attention = ScaledDotProduct(q, k, v, mask)                        # (batch, head, seq, features_by_head)
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))                           # (batch, seq, head, features_by_head)
        concat_output = tf.reshape(attn_output, (batch_size, seq_length, -1))           # (batch, seq, features)
        output = self.dense(concat_output)

        return output, attention


def get_ffn(d_model, hidden_size, act_func='relu'):
    return keras.models.Sequential([
                                  keras.layers.Dense(hidden_size, activation=act_func),
                                  keras.layers.Dense(d_model)
    ], name='SEQ')


class EncoderLayer(keras.layers.Layer):
    def __init__(self, dk=256, num_heads=8, hidden_layer_size=256, drop_rate=0.1):
        super(EncoderLayer, self).__init__()

        # layers
        self.MH = MultiHeadAttention(dk, num_heads)
        # self.ffn = get_ffn(dk, hidden_layer_size)
        self.normLayer1 = keras.layers.LayerNormalization(epsilon=1e-6)
        # self.normLayer2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(drop_rate)
        # self.dropout2 = keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        # multihead attention
        attn_output, _ = self.MH(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        z = self.normLayer1(x + attn_output)
        # normalization and feed forward layers
        # output = self.ffn(z)
        # output = self.dropout2(output, training=training)
        # output = self.normLayer2(z + output)

        return z


class DecoderLayer(keras.layers.Layer):
    def __init__(self, dk=256, num_heads=8, hidden_layer=256, drop_rate=0.1):
        super(DecoderLayer, self).__init__()
        # layers
        self.SAMH = MultiHeadAttention(dk, num_heads)
        self.EDMH = MultiHeadAttention(dk, num_heads)
        # self.ffn = get_ffn(dk, hidden_layer)

        self.normLayer1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.normLayer2 = keras.layers.LayerNormalization(epsilon=1e-6)
        # self.normLayer3 = keras.layers.LayerNormalization(epsilon=1e-6)\

        self.dropout1 = keras.layers.Dropout(drop_rate)
        self.dropout2 = keras.layers.Dropout(drop_rate)
        # self.dropout3 = keras.layers.Dropout(drop_rate)

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
        # output = self.ffn(z)
        # output = self.dropout3(output, training=training)
        # output = self.normLayer3(z + output)

        return z, self_attn, enc_dec_attn


class Encoder(keras.layers.Layer):
    def __init__(self, features_size, max_size, dk_model=256, num_heads=8, num_encoders=6,
                 enc_hidden_size=256, drop_rate=0.1):
        super(Encoder, self).__init__()

        # params
        self.dk_model = dk_model
        self.max_size = max_size
        self.enc_hidden_size = enc_hidden_size
        self.num_encoders = num_encoders

        # layers
        self.positional_encoding = positional_encoding(self.max_size, self.dk_model)
        self.embedding = keras.layers.Dense(dk_model)
        self.encoders_stack = [EncoderLayer(dk_model, num_heads, enc_hidden_size, drop_rate) for _ in
                               range(num_encoders)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, padding_mask, training):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dk_model, tf.float32))
        x += self.positional_encoding
        x = self.dropout(x, training=training)

        for encoder_layer in self.encoders_stack:
            x = encoder_layer(x, training, padding_mask)

        return x


class Decoder(keras.layers.Layer):
    def __init__(self, features_size, max_size, dk_model=256, num_heads=8, num_decoders=6,
                 dec_hidden_size=256, drop_rate=0.1):

        super(Decoder, self).__init__()

        # params
        self.dk_model = dk_model
        self.max_size = max_size
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
        x += self.positional_encoding
        x = self.dropout(x, training=training)

        for decoder_layer in self.decoders_stack:
            x, attn1, attn2, = decoder_layer(x, enc_output, training, look_ahead_mask, padding_mask)

        return x


class Transformer(keras.Model):
    def __init__(self, features_size, max_seq_size, dk=256,
                 enc_heads=8, dec_heads=8, num_encoders=6, num_decoders=6,
                 dec_hidden_size=256, drop_rate=0.1):

        super(Transformer, self).__init__()
        # layers
        self.encoder = Encoder(features_size, max_seq_size, dk, num_heads=enc_heads,
                               num_encoders=num_encoders)
        self.decoder = Decoder(features_size, max_seq_size, dk, num_heads=dec_heads,
                               num_decoders=num_decoders)
        self.linear = tf.keras.layers.Dense(80, name='Linear_Trans')

    def call(self, inputs, training, use_look_mask=True):
        inp, inp_masks, targets, tar_masks = inputs
        enc_out = self.encoder(inp, inp_masks, training)  # (batch, neighbors or sequence , attn dim , features)
        look_mask = get_look_ahead_mask(targets.shape[1]) if use_look_mask else None
        output = self.decoder(targets, enc_out, look_mask, tar_masks, training)
        output = self.linear(output)

        return output


class SemanticMapFeatures(keras.layers.Layer):
    def __init__(self, N, neighbors, out_dims, kernel_sizes, strides):
        super(SemanticMapFeatures, self).__init__()
        self.N = N
        self.neighbors = neighbors
        # self.ConvLayers = [keras.layers.Conv2D(out_dims[i], kernel_sizes[i], strides=strides[i], data_format='channels_first') for i in range(N)]
        # self.reshape = keras.layers.Reshape([-1, neighbors, 28 * 28])
        self.ConvLayers = []
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        h, w, c = 256, 256, 3
        for i in range(N):
            self.ConvLayers.append(
                keras.layers.Conv2D(out_dims[i], kernel_sizes[i], strides=strides[i]))
            h = (h - kernel_sizes[i]) // 2 + 1
            w = (w - kernel_sizes[i]) // 2 + 1
            c = out_dims[i]

    def call(self, inputs, **kwargs):
        output = inputs
        output = tf.reshape(output, [-1, 256, 256, 3])
        for layer in self.ConvLayers:
            output = layer(output)

        output = tf.keras.activations.tanh(output)
        output = tf.reshape(output, [-1, self.neighbors, 12 * 12])
        output = self.dense(output)
        return output


class STE_Transformer(keras.Model):
    def __init__(self, features_size, seq_size, neigh_size,
                 sp_dk=256, sp_enc_heads=8, sp_dec_heads=8, sp_num_encoders=6, sp_num_decoders=6,
                 tm_dk=256, tm_enc_heads=8, tm_dec_heads=8, tm_num_encoders=6, tm_num_decoders=6,
                 emb_size=64, drop_rate=0.1):
        super(STE_Transformer, self).__init__()

        self.emb_size = emb_size
        self.seq_size = seq_size
        self.neigh_size = neigh_size
        # layers
        self.semantic_map = SemanticMapFeatures(4, neigh_size, out_dims=[16, 16, 16, 1], kernel_sizes=[5, 5, 5, 7],
                                                strides=[2, 2, 2, 2])
        self.time_transformer = Transformer(features_size, seq_size, dk=tm_dk, enc_heads=tm_enc_heads,
                                            dec_heads=tm_dec_heads,
                                            num_encoders=tm_num_encoders, num_decoders=tm_num_decoders)

        # self.linear = tf.keras.layers.Dense(2, name='Linear_Trans')
        self.embeddings = tf.keras.layers.Dense(64, name='Embeddings')
        #self.spatial_mlp = tf.keras.layers.Dense(512)
        self.spatial_mlp = keras.models.Sequential([keras.layers.Dense(512, activation='relu'),
                                                    keras.layers.Dense(256)])
        self.offset = keras.layers.Dense(2)
        # training
        self.loss_object = tf.keras.losses.MeanSquaredError(reduction='sum')
        self.final_checkpoint = tf.train.Checkpoint(model=self)
        self.optimizer = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @tf.function
    def call(self, inputs, training, stds):
        """
          speeds.shape = (batch, neighbors, seq, feats)
          stds = tf.constant([[[[std_x, std_y]]]], dtype=tf.float32)
        """
        past, past_speed, past_seq_masks, past_neigh_masks, past_speed_masks = inputs[0]
        future, future_speed, future_seq_masks, futu_neigh_masks, futu_speed_masks = inputs[1]
        maps = inputs[2]

        squeezed_seq_mask = tf.squeeze(future_seq_masks)
        squeezed_neigh_mask = tf.squeeze(futu_neigh_masks)
        squeeze_past_neigh_mask = tf.squeeze(past_neigh_masks)
        squeezed_future_mask = tf.squeeze(futu_speed_masks)

        proc_maps = self.semantic_map(maps)
        # multiply by ones to match all neighbors shape, except features dim
        sp_desired_shape = past.shape[:-1] + proc_maps.shape[-1]
        sp_proc_maps = proc_maps[:, tf.newaxis, :, :] * tf.ones(sp_desired_shape)

        embeddings = self.embeddings(past)
        embeddings = tf.concat((embeddings, sp_proc_maps), axis=-1)
        embeddings = squeeze_past_neigh_mask[:, :, :, tf.newaxis] * embeddings      # (keep batch, keep seq, keep neigh, copy mask to feat dim)
        embeddings = tf.reshape(embeddings, [-1, self.seq_size, self.neigh_size * self.emb_size * 2])
        embeddings = self.spatial_mlp(embeddings)

        future_embeddings = self.embeddings(future)
        future_embeddings = tf.concat((future_embeddings, sp_proc_maps), axis=-1)
        future_embeddings = squeezed_neigh_mask[:, :, :, tf.newaxis] * future_embeddings
        future_embeddings = tf.reshape(future_embeddings, [-1, self.seq_size, self.neigh_size * self.emb_size * 2])
        future_embeddings = self.spatial_mlp(future_embeddings)

        output = self.time_transformer([embeddings, tf.squeeze(past_seq_masks)[:, tf.newaxis, tf.newaxis, :],
                                        future_embeddings, squeezed_seq_mask[:, tf.newaxis, tf.newaxis, :]], training)     # (batch, seq, features and neigh)
        # masking output
        output = tf.reshape(output, [-1, self.seq_size, self.neigh_size, 16])                                               # (batch, seq, neigh, feat)
        output = output[:, 1:, :, :] - output[:, :-1, :, :]
        output = self.offset(output)
        output = output * stds
        output = mask_output(output, squeezed_future_mask, 'seq')
        output = tf.concat([future[:, 0, :, :2][:, tf.newaxis, :, :], output], axis=1)
        output = tf.math.cumsum(output, axis=1)
        # output = tf.transpose(output, [0, 2, 1, 3])                   # (batch, seq, neigh, [x,y])
        output = mask_output(output, squeezed_neigh_mask, 'neigh')
        # output = self.linear(output)
        return output

    def loss_function(self, real, pred, neighbors_mask):
        # adapt mask and make mask shape match pred
        # neighbors_mask = 1 - neighbors_mask
        # neighbors_mask = neighbors_mask[:, :, :, np.newaxis]
        # pred_masked = pred * neighbors_mask
        pred_masked = pred
        loss_ = self.loss_object(real, pred_masked) * (1./(self.seq_size * self.neigh_size * 128))
        return loss_

    @tf.function
    def train_step(self, inputs):
        past, future, maps, stds = inputs
        # remove np.newaxis to match MultiHeadAttention
        neigh_out_masks = tf.squeeze(future[2])

        with tf.GradientTape() as tape:
            predictions = self((past, future, maps), True, stds)
            loss = self.loss_function(future[0][:, :, :, :2], predictions, neigh_out_masks)

        print('loss: ', loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def eval_step(self, past, future, maps, stds):
        preds = self((past, future, maps), False, stds)

        # transpose sequence with neigh dimension
        targets = tf.transpose(future[0][:, :, :, :2], [0, 2, 1, 3])
        preds = tf.transpose(preds, [0, 2, 1, 3])
        # reshape to remove batch
        targets = tf.reshape(targets, (-1, 26, 2))
        preds = tf.reshape(preds, (-1, 26, 2))

        return ADE(targets.numpy(), preds.numpy())

    def save_model(self, filepath='Code/weights/best_ModelTraj_weights'):
        self.final_checkpoint.write(filepath)

    def load_model(self, filepath='Code/weights/best_ModelTraj_weights'):
        self.final_checkpoint.restore(filepath)
