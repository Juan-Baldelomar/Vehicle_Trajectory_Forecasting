import numpy as np
from matplotlib import pyplot


def stamp_traj(inputs: np.ndarray, masks: np.ndarray, bitmaps: np.ndarray,
                              pixbymeter: float, yaw, step_start=-1, step_end=1, bottom=True):
    """
    :param: inputs    : np array of the form (S, N, F). S=sequence, N=Neighbors, F=Features (x=0, y=1).
    :param: masks     : np array of the form (S, N). S=sequence, N=Neighbors. Indicates thoses entries that are padded.
    :param: bitmaps   : np array of the form (L, H, W). L=Layers, H=Height, W=Width.
    :param: pixbymeter: relation of number of pixels by each meter
    :param: yaw       : rotation angle of the points (if map was rotated, trajectories should be rotated by the same angle)
    :param: step start: when stamping the step of and agent, a pixel width might be to small, so you can make it bigger by setting the step_start
                        and step_end, so you stamp all the pixels in the range (pos + step_start, pos + step_end)
    :param: bottom     : if True, stamp pixels in blue layer (RGB), else in red layer
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
    if bottom:
        neigh_bitmaps = np.append(neigh_bitmaps, stamped_positions[:, np.newaxis, :, :], axis=1)
    else:
        neigh_bitmaps = np.append(stamped_positions[:, np.newaxis, :, :], neigh_bitmaps, axis=1)

    return neigh_bitmaps
