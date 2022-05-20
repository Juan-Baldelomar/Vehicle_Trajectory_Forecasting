
import numpy as np
import tensorflow as tf


def stamp_positions_in_bitmap(inputs: np.ndarray, masks: np.ndarray, bitmaps: np.ndarray,
                              pixbymeter: float, yaw, step_start=-1, step_end=1, debug=False):
    """
    :param: inputs    : np array of the form (S, N, F). S=sequence, N=Neighbors, F=Features (x=0, y=1).
    :param: masks     : np array of the form (S, N). S=sequence, N=Neighbors. Indicates thoses entries that are padded.
    :param: bitmaps   : np array of the form (L, H, W). L=Layers, H=Height, W=Width.
    :param: pixbymeter: relation of number of pixels by each meter
    :param: yaw       : rotation angle of the points (if map was rotated, trajectories should be rotated by the same angle)
    :param: step start: when stamping the step of and agent, a pixel width might be to small, so you can make it bigger by setting the step_start
                        and step_end, so you stamp all the pixels in the range (pos + step_start, pos + step_end)
    :param: debug     : if True, print values of pixels
    """
    assert(step_start <= 0 and step_end >= 0)
    # needed shapes
    S, N, _ = inputs.shape
    n_layers, H, W = bitmaps.shape
    neigh_bitmaps = np.ones([N, n_layers, H, W]) * bitmaps[np.newaxis, :, :, :]
    # center pixels
    x_p, y_p = H / 2., W / 2.
    # get x, y for each observation, select the points that are not padded (masks==0) and get array flattened
    x = inputs[:, :, 0][masks == 0]
    y = inputs[:, :, 1][masks == 0]
    # perform rotation (clockwise)
    pix_x = x * np.cos(yaw) + y * np.sin(yaw)
    pix_y = -x * np.sin(yaw) + y * np.cos(yaw)
    # transform to pixel position
    pix_x = (pix_x * pixbymeter + x_p).astype(np.int32)
    pix_y = (pix_y * pixbymeter + y_p).astype(np.int32)
    pix_x = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_x))
    pix_y = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_y))
    # get neighbor dimension indexes, select the ones that are not padded and get array flattened
    N_pos = np.arange(N)[np.newaxis, :] * np.ones([S, N]).astype(np.int32)
    N_pos = N_pos[masks == 0]
    # build positions layers
    stamped_positions = np.zeros((N, H, W))
    # stamp values
    for i in range(step_start, step_end + 1):
        for j in range(step_start, step_end + 1):
            stamped_positions[(N_pos, pix_y + i, pix_x + j)] = 255.0
    # append new layer
    neigh_bitmaps = np.append(neigh_bitmaps, stamped_positions[:, np.newaxis, :, :], axis=1)
    if debug:
        print('x: ', np.reshape(pix_x, (S, N)))
        print('y: ', np.reshape(pix_y, (S, N)))

    return neigh_bitmaps


def stamp_positions_by_batch(inputs: np.ndarray, masks: np.ndarray, bitmaps: np.ndarray,
                             pixbymeter: float, yaw, step_start=-1, step_end=1, debug=False):
    """
    :param: inputs    : np array of the form (S, N, F). S=sequence, N=Batch * Neighbors, F=Features (x=0, y=1).
    :param: masks     : np array of the form (S, N). S=sequence, N=Batch * Neighbors. Indicates thoses entries that are padded.
    :param: bitmaps   : np array of the form (L, H, W). L=Layers, H=Height, W=Width.
    :param: pixbymeter: relation of number of pixels by each meter
    :param: yaw       : (B * N, S) rotation angle of the points (if map was rotated, trajectories should be rotated by the same angle)
    :param: step start: when stamping the step of and agent, a pixel width might be to small, so you can make it bigger by setting the step_start
                        and step_end, so you stamp all the pixels in the range (pos + step_start, pos + step_end)
    :param: debug     : if True, print values of pixels
    """
    assert(step_start <= 0 and step_end >= 0)
    # needed shapes
    S, N, _ = inputs.shape
    nxb, n_layers, H, W = bitmaps.shape
    neigh_bitmaps = bitmaps
    # center pixels
    x_p, y_p = H / 2., W / 2.
    # get x, y for each observation, select the points that are not padded (masks==0) and get array flattened
    x = inputs[:, :, 0][masks == 0]
    y = inputs[:, :, 1][masks == 0]
    # perform rotation (clockwise)
    pix_x = x * np.cos(yaw[masks == 0]) + y * np.sin(yaw[masks == 0])
    pix_y = -x * np.sin(yaw[masks == 0]) + y * np.cos(yaw[masks == 0])
    # transform to pixel position
    pix_x = (pix_x * pixbymeter + x_p).astype(np.int32)
    pix_y = (pix_y * pixbymeter + y_p).astype(np.int32)
    pix_x = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_x))
    pix_y = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_y))
    # get neighbor dimension indexes, select the ones that are not padded and get array flattened
    N_pos = np.arange(N)[np.newaxis, :] * np.ones([S, N]).astype(np.int32)
    N_pos = N_pos[masks == 0]
    # build positions layers
    # stamp values
    for i in range(step_start, step_end + 1):
        for j in range(step_start, step_end + 1):
            neigh_bitmaps[:, 2, :, :][(N_pos, pix_y + i, pix_x + j)] = 255.0
    # append new layer
    if debug:
        print('x: ', np.reshape(pix_x, (S, N)))
        print('y: ', np.reshape(pix_y, (S, N)))

    return neigh_bitmaps


def adapt_spa_mask(mask):
    return mask[np.newaxis, :, np.newaxis, :].astype(np.float32)   # (<new axis head>, seq, <new axis neighbor>, neighbors)
                                                                   #  to broadcast when doing addition in the attention layer


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

#@tf.function
def get_npz_bitmaps(path, past_xy, masks, yaw):
    img = np.load(path)
    bitmap = img['bitmaps']
    bitmap = stamp_positions_in_bitmap(past_xy, np.squeeze(masks), bitmap, 256 / 200.0, yaw)
    bitmap = np.transpose(bitmap, [0, 2, 3, 1])
    return bitmap.astype(np.float32)


AUTOTUNE = tf.data.experimental.AUTOTUNE


#@tf.function
def buildDataset(inputs, batch_size, pre_path=None, strategy: tf.distribute.MirroredStrategy=None):
    batch_size_per_replica = batch_size
    if strategy is not None:
        num_replicas = strategy.num_replicas_in_sync
        print('[MSG] NUM OF REPLICAS:', num_replicas)
        batch_size_per_replica //= num_replicas
 	
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
    past_seq_masks, future_seq_masks = [], []

    # get masks
    for input_ in inputs:
        past_speed_masks.append(adapt_seq_mask(input_['past_seqMask'])[:, :, :, 1:])
        futu_speed_masks.append(adapt_seq_mask(input_['future_seqMask'])[:, :, :, 1:])
        past_neigh_masks.append(adapt_spa_mask(input_['past_neighMask']))
        futu_neigh_masks.append(adapt_spa_mask(input_['future_neighMask']))
        past_seq_masks.append(adapt_seq_mask(input_['past_seqMask']))
        future_seq_masks.append(adapt_seq_mask(input_['future_seqMask']))
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
    future_ds = tf.data.Dataset.from_tensor_slices((future, future_speed, future_seq_masks, futu_neigh_masks, futu_speed_masks))
    target_ds = tf.data.Dataset.from_tensor_slices((future_shifted, full_traj, yaws))
    bitmaps_ds = tf.data.Dataset.from_tensor_slices((ids, past, past_neigh_masks, yaws))
    bitmaps_ds = bitmaps_ds.map(lambda id_, past_xy, masks, yaw: tf.numpy_function(func=get_npz_bitmaps,
                                                                                   inp=[id_, past_xy, masks, yaw],
                                                                                   Tout=tf.float32), num_parallel_calls=AUTOTUNE)

    #bitmaps_ds = bitmaps_ds.map(lambda x: tf.reshape(x, [5, 256, 256, 3]))
    # BUILD FINAL DATASET
    dataset = tf.data.Dataset.zip((past_ds, future_ds, bitmaps_ds, target_ds))
    #dataset = tf.data.Dataset.zip((past_ds, future_ds, target_ds))
    # SHUFFLE AND BATCH
    dataset = dataset.shuffle(1000)
    drop_remainder = len(past) % batch_size_per_replica == 1
    dataset = dataset.batch(batch_size_per_replica, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    #print(dataset.element_spec[2])
    if strategy is not None:
        dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset, std_x, std_y
