
import numpy as np
import tensorflow as tf

from InputQuery import stamp_positions_in_bitmap


def get_look_ahead_mask(input_data):
    input_shape = list(input_data.shape)[:-1]
    input_shape.insert(-1, input_shape[-1])
    input_shape.insert(1, 1)
    mask = 1 - tf.linalg.band_part(tf.ones(input_shape), -1, 0)
    return mask


def adapt_spa_mask(mask):
    return mask[np.newaxis, :, np.newaxis, :].astype(np.float32)   # (<new axis head>, seq, <new axis neighbor>, neighbors)
                                                                   # to broadcast when doing addition in the attention layer


def adapt_seq_mask(mask):
    return mask[np.newaxis, np.newaxis, np.newaxis, :].astype(np.float32)   # (1 (head), 1(neighbors), 1(seq), seq)


# normalize an image
def normalize(img):
    img = (img / 127.5) - 1
    return img


def get_img(image_path):
    raw_image = tf.io.read_file(image_path)
    raw_image = tf.image.decode_png(raw_image)
    raw_image = tf.image.resize(raw_image, (256, 256))

    # Convert both images to float32 tensors
    raw_image = tf.cast(tf.identity(raw_image), tf.float32)
    raw_image = normalize(raw_image)

    return raw_image


def get_npz_bitmaps(path, past_xy, masks, yaw):
    img = np.load(path)
    bitmap = img['bitmaps']
    bitmap = stamp_positions_in_bitmap(past_xy, np.squeeze(masks), bitmap, 256 / 200.0, yaw)
    bitmap = np.transpose(bitmap, [0, 2, 3, 1])
    return bitmap.astype(np.float32)


AUTOTUNE = tf.data.AUTOTUNE


def buildDataset(inputs, batch_size, origin_vals=None, pre_path=None):
    # get ids dataset
    if inputs[0]['ego_id'] is not None:
        ids = [pre_path + input_['ego_id'] + '.npz' for input_ in inputs]
        # imgs_dataset = ids_dataset.map(lambda x: tf.numpy_function(func=get_img, inp=[x], Tout=((tf.float32))), num_parallel_calls=AUTOTUNE)

    past = np.array([inputs[i]['past'] for i in range(len(inputs))])[:, :, :, :3].astype(np.float32)
    future = np.array([inputs[i]['future'] for i in range(len(inputs))])[:, :, :, :3].astype(np.float32)
    full_traj = np.array([inputs[i]['full_traj'] for i in range(len(inputs))])[:, :, :, :3].astype(np.float32)

    past_speed_masks, past_neigh_masks = [], []
    futu_speed_masks, futu_neigh_masks = [], []
    yaws = []
    past_seq_masks = []

    # get masks
    for input_ in inputs:
        past_speed_masks.append(adapt_seq_mask(input_['past_seqMask'])[:, :, :, 1:])
        futu_speed_masks.append(adapt_seq_mask(input_['future_seqMask'])[:, :, :, 1:])
        past_neigh_masks.append(adapt_spa_mask(input_['past_neighMask']))
        futu_neigh_masks.append(adapt_spa_mask(input_['future_neighMask']))
        past_seq_masks.append(input_['past_seqMask'].astype(np.float32))
        yaws.append(float(input_['origin_yaw']))

    # get each agent trajectory origin as 0, 0
    future_shifted = future[:, :, :, :2] - future[:, 0, :, :2][:, np.newaxis, :, :2]

    # past speeds normalized
    past_speed = past[:, 1:, :, 0:2] - past[:, :-1, :, 0:2]
    std_x, std_y = np.std(np.reshape(past_speed, (-1, 2)), axis=0)
    past_speed = past_speed / np.array([[std_x, std_y]])
    # future speeds normalized (by past)
    future_speed = future[:, 1:, :, 0:2] - future[:, :-1, :, 0:2]
    future_speed = future_speed / np.array([[std_x, std_y]])
    # transpose neigh and seq dims
    past_speed = np.transpose(past_speed, [0, 2, 1, 3])
    future_speed = np.transpose(future_speed, [0, 2, 1, 3])

    # get datasets
    past_ds = tf.data.Dataset.from_tensor_slices((past, past_speed, past_seq_masks, past_neigh_masks, past_speed_masks))
    future_ds = tf.data.Dataset.from_tensor_slices((future_shifted, future_speed, futu_neigh_masks, futu_speed_masks))
    target_ds = tf.data.Dataset.from_tensor_slices((future, full_traj, yaws))
    bitmaps_ds = tf.data.Dataset.from_tensor_slices((ids, past, past_neigh_masks, yaws))
    bitmaps_ds = bitmaps_ds.map(lambda id_, past_xy, masks, yaw: tf.numpy_function(func=get_npz_bitmaps,
                                                                                   inp=[id_, past_xy, masks, yaw],
                                                                                   Tout=tf.float32),
                                num_parallel_calls=AUTOTUNE)

    # BUILD FINAL DATASET
    dataset = tf.data.Dataset.zip((past_ds, future_ds, bitmaps_ds, target_ds))
    # SHUFFLE AND BATCH
    dataset = dataset.shuffle(4000)
    drop_remainder = len(past) % batch_size == 1
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)

    return dataset, std_x, std_y
