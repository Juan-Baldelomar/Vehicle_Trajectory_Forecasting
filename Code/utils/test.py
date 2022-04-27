import numpy as np

from nuscenes_dataloader import NuscenesLoader
from shifts_dataloader import ShiftsLoader
from InputQuery import *
from InputQuery import verifyNan
from InputQuery import process_nans
import dataloader as dl


# PATH
dataroot_base = '/data/sets/nuscenes'
dataroot_train = '/media/juan/Elements'

# dataset attributes
dataroot = dataroot_train + dataroot_base
#dataroot = dataroot_base

# VERSION
version = 'v1.0-trainval'
#version = 'v1.0-mini'

# NAME
data_name = 'train'
#data_name = 'mini_train'


# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name, loadMap=False)
#
# inputQuery = InputQuery(nuscenes_loader)
#
# matrixes, ids = inputQuery.get_single_Input(7, 8, 7)


#dl.save_pkl_data(matrixes, 'nusc_inps.pkl')

# ------------------------------------------------------------ multiple agents by scene ------------------------------------------------------------------
# nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name, loadMap=True)
# inputQuery = InputQuery(nuscenes_loader)
#
# store = False
#
# cubes, ids = inputQuery.get_TransformerCube_Input(8, 7, 10, offset=7, get_maps=None, path='maps/masks', maps=nuscenes_loader.maps)
# agent_cubes, agent_ids = inputQuery.get_input_ego_change(8, 7, 10, offset=7, get_maps=None, maps=nuscenes_loader.maps)
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
dataroot = '/data/shifts/data/development'
shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=True)
shifts_loader.dataset.get_ego_indexes(50)
inputQuery = InputQuery(shifts_loader)

ego_iter = iter(shifts_loader.dataset.ego_vehicles.items())
ego_id, ego_vehicle = next(ego_iter)

shifts_bitmap = ShiftsBitmap()

inps = inputQuery.get_egocentered_input(ego_vehicle, shifts_loader.dataset.agents, 50, 5, 0,
                                        24, bitmap_extractor=shifts_bitmap)

yaw = list(ego_vehicle.timesteps.values())[24].rot #* 180.0/np.pi
bitmaps = stamp_positions_in_bitmap(inps[0], inps[2], 512/200.0, yaw)
bitmaps = np.transpose(bitmaps, [0, 2, 3, 1])

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

plt.figure(figsize=(10, 10))
plt.imshow(bitmaps[0])
plt.show()
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