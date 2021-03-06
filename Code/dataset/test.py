import numpy as np

from nuscenes_dataloader import NuscenesLoader
from shifts_dataloader import ShiftsLoader
from InputQuery import *
from InputQuery import verifyNan
from InputQuery import process_nans
from Code.dataset.dataset import stamp_positions_in_bitmap, buildDataset
from Code.utils import save_utils as dl

# PATH
dataroot_base = '/data/sets/nuscenes'
dataroot_train = '/media/juan/Elements'

# dataset attributes
dataroot = dataroot_train + dataroot_base
# dataroot = dataroot_base

# VERSION
version = 'v1.0-trainval'
# version = 'v1.0-mini'

# NAME
data_name = 'train'
# data_name = 'mini_train'


# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name, loadMap=False)
#
# inputQuery = InputQuery(nuscenes_loader)
#
# matrixes, ids = inputQuery.get_single_Input(7, 8, 7)


# dl.save_pkl_data(matrixes, 'nusc_inps.pkl')

# ------------------------------------------------------------ multiple agents by scene ------------------------------------------------------------------
# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name, loadMap=True)
# inputQuery = InputQuery(nuscenes_loader)
#
# store = False
#
# nuscenes_bitmap = NuscenesBitmap(nuscenes_loader.maps)
# cubes, ids = inputQuery.get_TransformerCube_Input(8, 7, 10, offset=7, bitmap_extractor=nuscenes_bitmap, path='maps/masks')
# agent_cubes, agent_ids = inputQuery.get_input_ego_change(8, 7, 10, offset=7, bitmap_extractor=nuscenes_bitmap)
#
# if store:
#     pass
#     # remove nans
#     #process_nans(cubes)
#     #process_nans(agent_cubes)
#     #verifyNan(cubes, ids)
#     #verifyNan(agent_cubes, agent_ids)
#     # store information
#     #final_cubes = cubes + agent_cubes
#     #final_ids = ids + agent_ids
#     #dl.save_pkl_data({'inputs': final_cubes, 'ids': final_ids}, 'nusc_multiple_agents_inp.pkl')
#
# #nuscenes_loader.nuscenes.get('sample_annotation', '67359ca5094147f3b3b210d406873407')
#
# # count scenes that do have neighbors with a trajectory to forecast
# # count = 0
# # for vehicle in nuscenes_loader.dataset['ego_vehicles'].values():
# #     for timestep in vehicle.values():
# #         if len(timestep['neighbors']) > 0:
# #             count += 1
# #             break

# -------------------------------------------------------------------- SHIFTS --------------------------------------------------------------------
# dataroot = '/data/shifts/data/development'
# chunk_number = 10
# pickle_filename = '/data/shifts/data_chunk' + str(chunk_number) + '.pkl'
# maps_path = '../data/maps/shifts/chunk' + str(chunk_number)
# final_path = '../data/shifts_data_chunk' + str(chunk_number) + '.pkl'
# chunk_start = (chunk_number-1) * 2000
# chunk_end = chunk_number * 2000
# shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=True, pickle_filename=pickle_filename, chunk=(chunk_start, chunk_end))
# inputQuery = InputQuery(shifts_loader)
# shifts_bitmap = ShiftsBitmap()
# inputs = inputQuery.get_TransformerCube_Input(25, 25, 5, 24, bitmap_extractor=None, path=maps_path)
#dl.save_pkl_data(inputs, final_path)


# inputs_1 = dl.load_pkl_data('../data/shifts_data_chunk1.pkl')
# inputs_2 = dl.load_pkl_data('../data/shifts_data_chunk2.pkl')
# inputs_3 = dl.load_pkl_data('../data/shifts_data_chunk3.pkl')
# inputs_4 = dl.load_pkl_data('../data/shifts_data_chunk4.pkl')
# inputs_5 = dl.load_pkl_data('../data/shifts_data_chunk5.pkl')
# inputs_6 = dl.load_pkl_data('../data/shifts_data_chunk6.pkl')
# inputs_7 = dl.load_pkl_data('../data/shifts_data_chunk7.pkl')
# inputs_8 = dl.load_pkl_data('../data/shifts_data_chunk8.pkl')
# inputs_9 = dl.load_pkl_data('../data/shifts_data_chunk9.pkl')
# inputs_10 = dl.load_pkl_data('../data/shifts_data_chunk10.pkl')
#
# all_inputs = inputs_1 + inputs_2 + inputs_3 + inputs_4 + inputs_5 + inputs_6 + inputs_7 + inputs_8 + inputs_9 + inputs_10
# dl.save_pkl_data(all_inputs, '../data/shifts_data_all_p4.pkl', protocol=4)


# --------------------------------------- ******* TEST visualize maps and stamped positions ******* ---------------------------------------
#file_id = inputs[1225]['ego_id'] + '.npz'
#ego_id, _ = file_id.split('_')
#ego_vehicle = shifts_loader.dataset.ego_vehicles[ego_id]

# img_comp = np.load('../data/maps/shifts/chunk10/' + file_id)
# img = img_comp['bitmaps']
# #img = np.transpose(img, [1, 2, 0])
# plt.figure(figsize=(10, 10))
# plt.imshow(img[:, :, 1])
# plt.show()
#
#inps = inputQuery.get_egocentered_input(ego_vehicle, shifts_loader.dataset.agents, 50, 5, 0,
#                                        24, bitmap_extractor=shifts_bitmap)

#
# yaw = list(ego_vehicle.timesteps.values())[24].rot
# bitmaps = stamp_positions_in_bitmap(inps[0][0:25], inps[1][0:25], inps[2], 256 / 200.0, yaw)
# bitmaps = np.transpose(bitmaps, [0, 2, 3, 1])
# plt.figure(figsize=(10, 10))
# plt.imshow(bitmaps[0])
# plt.show()

# ------------------------------------------------------------------------- TEST -------------------------------------------------------------------------

# # PATH
# dataroot_base = '/data/sets/nuscenes'
# dataroot_train = '/media/juan/Elements'
#
# # dataset attributes
# dataroot = dataroot_train + dataroot_base
# #dataroot = dataroot_base
#
# # VERSION
# version = 'v1.0-trainval'
# #version = 'v1.0-mini'
#
# # NAME
# data_name = 'val'
# #data_name = 'mini_train'
#
# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name, loadMap=False)
#
# inputQuery = InputQuery(nuscenes_loader)
#
# matrixes, ids = inputQuery.get_single_Input(7, 8, 7)
#
# dl.save_pkl_data(matrixes, 'val_nusc_inps.pkl')


import tensorflow as tf
data = dl.load_pkl_data('../data/shifts_data_chunk10.pkl')
dataset = buildDataset(data, 8, pre_path='../data/maps/shifts/chunk10/')

data_iter = iter(dataset[0])
past, future, maps, target = next(data_iter)

past_inp = target[0][:, 1:, :, :]
past_mask = tf.squeeze(future[2])[:, 1:, :]
yaws = target[2]
yaws = yaws[:, tf.newaxis, tf.newaxis] * tf.ones([8, 25, 5])
yaws = tf.transpose(yaws, [0, 2, 1])
yaws = tf.reshape(yaws, [-1, 25])
yaws = tf.transpose(yaws, [1, 0])

past_inp = tf.transpose(past_inp, [0, 2, 1, 3])
past_mask = tf.transpose(past_mask, [0, 2, 1])

past_inp = tf.reshape(past_inp, [-1, 25, 3])
past_mask = tf.reshape(past_mask, [-1, 25])

past_inp = tf.transpose(past_inp, [1, 0, 2])
past_mask = tf.transpose(past_mask, [1, 0])

#neigh_bitmaps = np.ones([8, 5, 2, 256, 256]) * maps[:, np.newaxis, :, :, :]
neigh_bitmaps = tf.transpose(maps, [0, 1, 4, 2, 3])
neigh_bitmaps = tf.reshape(neigh_bitmaps, [-1, 3, 256, 256])

bitmaps = stamp_positions_in_bitmap(past_inp.numpy(), past_mask.numpy(), neigh_bitmaps.numpy(), 256.0/200, yaws.numpy(), mode=2)
#bitmaps = bitmaps.numpy()
bitmaps = np.reshape(bitmaps, [-1, 5, 3, 256, 256])
#bitmaps_final_layer = bitmaps[:, :, 2, :, :] + bitmaps[:, :, 3, :, :]

#bitmaps[:, :, 2, :, :] = bitmaps_final_layer
bitmaps = np.transpose(bitmaps, [0, 1, 3, 4, 2])

plt.figure(figsize=(10, 10))
plt.imshow(bitmaps[1, 0, :, :, :3])
plt.show()

plt.imshow(maps[1, 0, :, :, :3])
plt.show()

