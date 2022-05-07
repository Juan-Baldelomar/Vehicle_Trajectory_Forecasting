import numpy as np

from nuscenes_dataloader import NuscenesLoader
from shifts_dataloader import ShiftsLoader
from InputQuery import *
from InputQuery import verifyNan
from InputQuery import process_nans
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
# chunk_number = 3
# pickle_filename = '/data/shifts/data_chunk' + str(chunk_number) + '.pkl'
# maps_path = '../data/maps/shifts/chunk' + str(chunk_number)
# final_path = '../data/shifts_data_chunk' + str(chunk_number) + '.pkl'
# chunk_start = (chunk_number-1) * 2000
# chunk_end = chunk_number * 2000
# shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=False, pickle_filename=pickle_filename, chunk=(chunk_start, chunk_end))
# inputQuery = InputQuery(shifts_loader)
# shifts_bitmap = ShiftsBitmap()
# inputs = inputQuery.get_TransformerCube_Input(25, 25, 5, 24, bitmap_extractor=None, path=maps_path)
# dl.save_pkl_data(inputs, final_path)


inputs_1 = dl.load_pkl_data('../data/shifts_data_chunk1.pkl')
inputs_2 = dl.load_pkl_data('../data/shifts_data_chunk2.pkl')
inputs_3 = dl.load_pkl_data('../data/shifts_data_chunk3.pkl')
inputs_4 = dl.load_pkl_data('../data/shifts_data_chunk4.pkl')
inputs_5 = dl.load_pkl_data('../data/shifts_data_chunk5.pkl')

all_inputs = inputs_1 + inputs_2 + inputs_3 + inputs_4 + inputs_5
dl.save_pkl_data(all_inputs, '../data/shifts_data_all_p4.pkl', protocol=4)

#dl.save_pkl_data({'inputs': inputs, 'ids': ids, 'yaws': yaws}, 'shifts_data.pkl')

# ******* TEST visualize maps and stamped positions *******
#ego_iter = iter(shifts_loader.dataset.ego_vehicles.items())
#ego_id, ego_vehicle = next(ego_iter)

# file_id = '47762072c090c2cfdb4123d28225f935_0.npz'
# ego_keys = list(shifts_loader.dataset.ego_vehicles.keys())
# ego_vehicle = shifts_loader.dataset.ego_vehicles['47762072c090c2cfdb4123d28225f935']
# ego_vehicle = shifts_loader.dataset.ego_vehicles[ego_keys[355]]
#
# img_comp = np.load('maps/shifts/47762072c090c2cfdb4123d28225f935_0.npz')
# img = img_comp['bitmaps']
# img = np.transpose(img, [1, 2, 0])
# plt.figure(figsize=(10, 10))
# plt.imshow(img[:, :, 1])
# plt.show()
#
# inps = inputQuery.get_egocentered_input(ego_vehicle, shifts_loader.dataset.agents, 50, 5, 0,
#                                         24, bitmap_extractor=shifts_bitmap)
#
# #
# yaw = list(ego_vehicle.timesteps.values())[24].rot
# bitmaps = stamp_positions_in_bitmap(inps[0], inps[1], inps[2], 256 / 200.0, yaw)
# bitmaps = np.transpose(bitmaps, [0, 2, 3, 1])
# plt.figure(figsize=(10, 10))
# plt.imshow(bitmaps[0])
# plt.show()

# v_track = VehicleTrack()
# origin = list(ego_vehicle.timesteps.values())[24]
# v_track.position.x = origin.x
# v_track.position.y = origin.y
# v_track.yaw = origin.rot
# v_track.linear_velocity.x = origin.x_speed
#
# positions = stamp_by_hand(inps[0], v_track)
# plt.figure(figsize=(10, 10))
# plt.imshow(positions)
# plt.show()
# plt.figure(figsize=(10, 10))
#
# plt.imshow(bitmaps[0][0], origin='lower', cmap='binary', alpha=1.0)
# #plt.imshow(out['feature_maps'][4], origin='lower', cmap='binary', alpha=0.7)
# plt.imshow(bitmaps[0][1], origin='lower', cmap='binary', alpha=0.5)
# plt.imshow(bitmaps[0][2], origin='lower', cmap='binary', alpha=0.3)


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
