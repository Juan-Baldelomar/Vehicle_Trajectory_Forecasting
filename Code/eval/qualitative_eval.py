import numpy as np
import os
import glob
from utils.save_utils import save_pkl_data

from utils import save_utils as dl
from dataset.DataModel import ShiftsBitmap
from dataset.DataModel import ShiftsEgoStep
from matplotlib import pyplot as plt


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
    bitmaps[:, 0, :, :] *= 0.5**2
    # needed shapes
    S, N, _ = inputs.shape
    _, n_layers, H, W = bitmaps.shape
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
    value = 2 if bottom else 0
    channel = np.full(len(N_pos), value) 
    # stamp values
    for i in range(step_start, step_end + 1):
        for j in range(step_start, step_end + 1):
            bitmaps[(N_pos, channel, pix_y + i, pix_x + j)] = 255.0

    return bitmaps


def draw_circle(x_c, y_c, radius):
    theta = np.linspace(0, 2 * np.pi, 100)
    # Setting radius
    # Generating x and y data
    x = radius * np.cos(theta) + x_c
    y = radius * np.sin(theta) + y_c
    return x, y


def draw_traj(x, y, pixbymeter, yaw, bitmap, length=2, width=2):
    length = length // 2
    width = width // 2
    H, W, _ = bitmap.shape
    # center pixels
    x_p, y_p = H / 2., W / 2.
    # perform rotation (clockwise)
    pix_x = x * np.cos(yaw) + y * np.sin(yaw)
    pix_y = -x * np.sin(yaw) + y * np.cos(yaw)
    # transform to pixel position
    pix_x = (pix_x * pixbymeter + x_p).astype(np.int32)
    pix_y = (pix_y * pixbymeter + y_p).astype(np.int32)
    if 0 < pix_x < W and 0 < pix_y < H:
        bitmap[pix_y :pix_y + 1, pix_x :pix_x + 1, 0] = 255.0
        bitmap[pix_y:pix_y + 1, pix_x:pix_x + 1, 1] = 255.0

    return bitmap


def draw_car(x, y, pixbymeter, yaw, bitmap, is_objective=False, draw_attn=True, length=5, width=5):
    length = length//2
    width = width//2
    H, W, _ = bitmap.shape
    # center pixels
    x_p, y_p = H / 2., W / 2.
    # perform rotation (clockwise)
    pix_x = x * np.cos(yaw) + y * np.sin(yaw)
    pix_y = -x * np.sin(yaw) + y * np.cos(yaw)
    # transform to pixel position
    pix_x = (pix_x * pixbymeter + x_p).astype(np.int32)
    pix_y = (pix_y * pixbymeter + y_p).astype(np.int32)
    if 0 < pix_x < W and 0 < pix_y < H:
        bitmap[pix_y - width:pix_y + width, pix_x - length:pix_x + length, 2] = 255.0
        if is_objective:
            bitmap[pix_y - width:pix_y + width, pix_x - length:pix_x + length, 1] = 255.0

    radius = max(length, width)
    radius = radius+2 if not is_objective else radius-1
    circle_x, circle_y = draw_circle(pix_x, pix_y, radius)
    for x_c, y_c in zip(circle_x, circle_y):
        x_c, y_c = int(x_c), int(y_c)
        if 0 < x_c < W and 0 < y_c < H and (draw_attn or is_objective):
            bitmap[y_c, x_c, 0] = 255.0

    return bitmap


def get_visual_attn(attention_inputs, masks):
    interest_points = []
    for num_input, (sample, mask) in enumerate(zip(attention_inputs, masks)):
        for n_head, head in enumerate(sample):
            for timestep in range(len(head)):
                if np.sum(mask[timestep]) >= 4:
                    continue
                for i in range(5):
                    # skip padded elements or
                    if mask[timestep, i] == 1:
                        continue
                    for j in range(5):
                        if mask[timestep, j] == 1:
                            continue
                        if i != j and sample[n_head, timestep, i, j] > 0.3:
                            interest_points.append((num_input, n_head, timestep))
    return interest_points


def process_attn(file):
    data = np.load(file)
    data.allow_pickle = True
    weights = data['weights'].reshape(-1, 8, 26, 5, 5)
    masks = np.squeeze(data['masks']).reshape(-1, 26, 5)
    points = get_visual_attn(weights, masks)
    return points


def visualize_sample(num_sample, np_points, masks, weights, ids_name, origins_info,
                     past_data_np, bitmap_gen, c_color=1.0, l=5, w=5, resolution=0.2):
    n_inp, head, timestep = np_points[num_sample]
    mask = masks[n_inp, timestep]
    weight_matrix = weights[n_inp, head, timestep]
    origin = origins_info[ids_name[n_inp]]
    step = ShiftsEgoStep(*origin[0])
    bitmaps = bitmap_gen.getMasks(step, map_name=origin[1])
    #

    bitmaps = np.transpose(bitmaps, [1, 2, 0])
    H, W, _ = bitmaps.shape
    zeros = np.zeros((H, W, 1))
    bitmaps = np.concatenate([bitmaps, zeros], axis=2)
    bitmaps = bitmaps * c_color
    for n_neigh, mask_val in enumerate(mask):
        if mask_val == 0:
            bitmaps = draw_car(*past_data_np[n_inp, timestep, n_neigh, 0:2], 1/resolution, origin[0][2],
                               bitmaps, n_neigh == 0, weight_matrix[0, n_neigh] > 0.5, length=l, width=w)

    for xt, yt in past_data_np[n_inp, :25, 0, 0:2]:
        bitmaps = draw_traj(xt, yt, 1/resolution, origin[0][2], bitmaps, 1, 2)

    plt.imshow(bitmaps)
    ax = plt.gca()
    # hide x-axis
    ax.get_xaxis().set_visible(False)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    plt.show()


#np_points = process_attn('../../attn_weights.npz')
#save_pkl_data(np_points, 'interest_points.pkl')

np_points = np.load('interest_points.npy')

# get ids
data = np.load('../../attn_weights.npz')
data.allow_pickle = True
ids = data['ids']
masks = np.squeeze(data['masks']).reshape(-1, 26, 5)
weights = data['weights'].reshape(-1, 8, 26, 5, 5)

ids = ids.reshape(-1)
ids_name = list(map(os.path.basename, ids))
ids_name = [name.decode('utf-8') for name in ids_name]
ids_name = [name.split('_')[0] for name in ids_name]

origins_info = dl.load_pkl_data('../dataset/origins_info.pkl')
#


# plot example
origin = origins_info[ids_name[0]]
step = ShiftsEgoStep(*origin[0])
#bitmaps = bitmap_gen.getMasks(step, map_name=origin[1])
#

#bitmaps = np.transpose(bitmaps, [1, 2, 0])
#zeros = np.zeros((128, 128, 1))
#bitmaps = np.concatenate([bitmaps, zeros], axis=2)
#
#plt.imshow(bitmaps)
#plt.show()

past_data_np = np.load('past_data.npy')


#car_ = draw_car(*past_data_np[0, 8, 0, 0:2], 0.2, origin[0][2], bitmaps)
#plt.imshow(car_)


resolution = 1.0
bitmap_gen = ShiftsBitmap(rows=128, cols=128, resolution=resolution)
n_inp = np.random.choice(len(np_points))
visualize_sample(n_inp, np_points, masks, weights, ids_name, origins_info, past_data_np, bitmap_gen, 0.5, 4, 6, resolution)

n_inp, head, timestep = np_points[n_inp]

np.transpose(past_data_np[n_inp, :, :, :2], [1, 0, 2])
weights[n_inp, head, timestep]
masks[n_inp]
# ---------------------------------- uncoment to visualize trajectories ----------------------------------
# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#
#     def visualize(index):
#         files = glob.glob('../qual_eval/*')
#         if index == -1:
#             start = 0
#             end = len(files)
#         else:
#             start = index
#             end = index + 1
#
#         all_bitmaps = [(file, np.load(file)['bitmaps']) for file in files[start:end]]
#         for name, bitmaps in all_bitmaps:
#             for bitmap in bitmaps:
#                 plt.imshow(np.transpose(-0.2 + bitmap, [1, 2, 0]))
#                 #plt.title(name)
#                 #plt.axis = False
#                 plt.show()
#     visualize(-1)
