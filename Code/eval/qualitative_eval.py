import numpy as np
import glob
#from matplotlib import pyplot


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


# files = glob.glob('../qual_eval/*')
# files = [f.split('/')[-1] for f in files]
# files = [f.split('_')[0] for f in files]
#
# from Code.utils import save_utils as dl
# from Code.dataset.DataModel import ShiftsBitmap
# from Code.dataset.DataModel import ShiftsEgoStep
#
# origins_info = dl.load_pkl_data('../dataset/origins_info.pkl')
#
# bitmap_gen = ShiftsBitmap(rows=512, cols=512, resolution=0.2)
# origin = origins_info[files[0]]
# step = ShiftsEgoStep(*origin[0])
# bitmaps = bitmap_gen.getMasks(step, map_name=origin[1])
#
# from matplotlib import pyplot as plt
# bitmaps = np.transpose(bitmaps, [1, 2, 0])
# zeros = np.zeros((512, 512, 1))
# bitmaps = np.concatenate([bitmaps, zeros], axis=2)
#
# plt.imshow(bitmaps)
# plt.show()
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    def visualize(index):
        files = glob.glob('../qual_eval/*')
        if index == -1:
            start = 0
            end = len(files)
        else:
            start = index
            end = index + 1

        all_bitmaps = [(file, np.load(file)['bitmaps']) for file in files[start:end]]
        for name, bitmaps in all_bitmaps:
            for bitmap in bitmaps:
                plt.imshow(np.transpose(-0.2 + bitmap, [1, 2, 0]))
                #plt.title(name)
                #plt.axis = False
                plt.show()
    visualize(-1)
