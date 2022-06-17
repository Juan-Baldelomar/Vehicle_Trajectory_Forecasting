import numpy as np

from nuscenes_dataloader import NuscenesLoader
from shifts_dataloader import ShiftsLoader
from InputQuery import *
from InputQuery import verifyNan
from InputQuery import process_nans
from Code.dataset.dataset import stamp_positions_in_bitmap, buildDataset
from Code.utils import save_utils as dl
#from Code.models.Model_traj import STTransformer
from Code.utils.save_utils import load_pkl_data

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
data_name = 'val'
# data_name = 'mini_train'


# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name, loadMap=False)
#
# inputQuery = InputQuery(nuscenes_loader)
#
# matrixes, ids = inputQuery.get_single_Input(7, 8, 7)


# dl.save_pkl_data(matrixes, 'nusc_inps.pkl')

# ------------------------------------------------------------ multiple agents by scene ------------------------------------------------------------------
#nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name, loadMap=True)
#inputQuery = InputQuery(nuscenes_loader)
#
# store = False
#
#nuscenes_bitmap = NuscenesBitmap(nuscenes_loader.maps)
#cubes, ids = inputQuery.get_TransformerCube_Input(8, 7, 10, offset=7, bitmap_extractor=nuscenes_bitmap, path='maps/masks')

#agent_cubes, agent_ids = inputQuery.get_egocentered_input(8, 7, 10, offset=7, bitmap_extractor=nuscenes_bitmap)
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

# --------------------------------------- ******* TEST visualize maps and stamped positions ******* ---------------------------------------
nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name, loadMap=True)
inputQuery = InputQuery(nuscenes_loader)

nuscenes_bitmap = NuscenesBitmap(nuscenes_loader.maps)
ego_ids = list(nuscenes_loader.dataset.ego_vehicles.keys())
ego_vehicle = nuscenes_loader.dataset.ego_vehicles[ego_ids[0]]

inputs = inputQuery.get_TransformerCube_Input(10, 12, 5, 9)

# nuscenes_loader.dataset.get_trajectories_indexes(use_ego_vehicles=True, L=40)
# nuscenes_bitmap = NuscenesBitmap(nuscenes_loader.maps)
# inps = inputQuery.get_egocentered_input(ego_vehicle, nuscenes_loader.dataset.agents, 40, 5, 0, 24, bitmap_extractor=nuscenes_bitmap)
#
# #
# yaw = list(ego_vehicle.timesteps.values())[24].rot
# bitmaps = stamp_positions_in_bitmap(inps[0][0:25], inps[1][0:25], inps[2], 1., yaw)
# bitmaps = np.transpose(bitmaps, [0, 2, 3, 1])
# #plt.figure(figsize=(5, 5))
# plt.imshow(bitmaps[0])
# plt.show()


# dataroot = '/data/shifts/data/train'
# shifts_pickle_filename = '/data/shifts/val/data_chunk1.pkl'
# datas = load_pkl_data(shifts_pickle_filename)
# shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=True, pickle_filename=shifts_pickle_filename)
# shifts_inputQuery = InputQuery(shifts_loader)
# shifts_bitmap = ShiftsBitmap()
#
# ego_ids = list(shifts_loader.dataset.ego_vehicles.keys())
# ego_vehicle = shifts_loader.dataset.ego_vehicles[ego_ids[0]]
#
# shifts_loader.dataset.get_trajectories_indexes(use_ego_vehicles=True, L=40)
# shifts_bitmap = ShiftsBitmap()
# inps = shifts_inputQuery.get_egocentered_input(ego_vehicle, shifts_loader.dataset.agents, 40, 5, 0, 24, bitmap_extractor=shifts_bitmap)
#
# yaw = list(ego_vehicle.timesteps.values())[24].rot
# bitmaps = stamp_positions_in_bitmap(inps[0][0:25], inps[1][0:25], inps[2], 256. / 200.0, yaw)
# bitmaps = np.transpose(bitmaps, [0, 2, 3, 1])
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


# import tensorflow as tf
# data = dl.load_pkl_data('../data/shifts_data_chunk10.pkl')
# dataset = buildDataset(data, 8, pre_path='../data/maps/shifts/chunk10/')
#
# data_iter = iter(dataset[0])
# past, future, maps, target = next(data_iter)
#
# past_inp = target[0][:, 1:, :, :]
# past_mask = tf.squeeze(future[2])[:, 1:, :]
# yaws = target[2]
# yaws = yaws[:, tf.newaxis, tf.newaxis] * tf.ones([8, 25, 5])
# yaws = tf.transpose(yaws, [0, 2, 1])
# yaws = tf.reshape(yaws, [-1, 25])
# yaws = tf.transpose(yaws, [1, 0])
#
# past_inp = tf.transpose(past_inp, [0, 2, 1, 3])
# past_mask = tf.transpose(past_mask, [0, 2, 1])
#
# past_inp = tf.reshape(past_inp, [-1, 25, 3])
# past_mask = tf.reshape(past_mask, [-1, 25])
#
# past_inp = tf.transpose(past_inp, [1, 0, 2])
# past_mask = tf.transpose(past_mask, [1, 0])
#
# #neigh_bitmaps = np.ones([8, 5, 2, 256, 256]) * maps[:, np.newaxis, :, :, :]
# neigh_bitmaps = tf.transpose(maps, [0, 1, 4, 2, 3])
# neigh_bitmaps = tf.reshape(neigh_bitmaps, [-1, 3, 256, 256])
#
# bitmaps = stamp_positions_in_bitmap(past_inp.numpy(), past_mask.numpy(), neigh_bitmaps.numpy(), 256.0/200, yaws.numpy(), mode=2)
# #bitmaps = bitmaps.numpy()
# bitmaps = np.reshape(bitmaps, [-1, 5, 3, 256, 256])
# #bitmaps_final_layer = bitmaps[:, :, 2, :, :] + bitmaps[:, :, 3, :, :]
#
# #bitmaps[:, :, 2, :, :] = bitmaps_final_layer
# bitmaps = np.transpose(bitmaps, [0, 1, 3, 4, 2])
#
# plt.figure(figsize=(10, 10))
# plt.imshow(bitmaps[1, 0, :, :, :3])
# plt.show()
#
# plt.imshow(maps[1, 0, :, :, :3])
# plt.show()
#

# past = np.random.rand(32, 26, 5, 3)
# future = np.random.rand(32, 26, 5, 3)
#
# past_s = np.random.rand(32, 5, 25, 3)
# future_s = np.random.rand(32, 5, 25, 3)
#
#
# mask1 = np.random.choice(1, [32, 1, 5, 1, 25]).astype('float32')
# mask2 = np.random.choice(1, [32, 1, 26, 1, 5]).astype('float32')
# mask3 = np.random.choice(1, [32, 1, 26, 1, 5]).astype('float32')
#
# mask1[0, 0, 0, 0, 5] = 1.
#
# maps = np.random.rand(32, 5, 256, 256, 3)
#
#
# model = STTransformer(32, 25, 5, 32, 1, 1, 1, 1, 32, 1, 1, 1, 1, 16)
#
# out = model(([past, past_s, mask3, mask2, mask1], [future, future_s, mask3, mask2, mask1], maps), True, [1, 2])
#
# attn = out[1][0]
#
# attn_0 = attn[0]
#
# attn_0 = np.squeeze(attn_0)

