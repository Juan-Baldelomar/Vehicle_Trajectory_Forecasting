import numpy as np

from nuscenes_dataloader import NuscenesLoader
from InputQuery import InputQuery
from InputQuery import verifyNan
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

nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=True, version=version, data_name=data_name)

inputQuery = InputQuery(nuscenes_loader)

cubes, ids = inputQuery.get_TransformerCube_Input(8, 7, 10, offset=7)
agent_cubes, agent_ids = inputQuery.get_input_ego_change(8, 7, 10, offset=7)

verifyNan(cubes, ids)
verifyNan(agent_cubes, agent_ids)

final_cubes = cubes + agent_cubes
dl.save_pkl_data(final_cubes, 'nusc_inps.pkl')


# count scenes that do have neighbors with a trajectory to forecast
# count = 0
# for vehicle in nuscenes_loader.dataset['ego_vehicles'].values():
#     for timestep in vehicle.values():
#         if len(timestep['neighbors']) > 0:
#             count += 1
#             break

# -------------------------------------------------------------------- MAPA --------------------------------------------------------------------


# This is the path where you stored your copy of the nuScenes dataset.
#DATAROOT = dataroot

# nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
# helper = PredictHelper(nuscenes)
# mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
# toks = [x.split('_') for x in mini_train]

#nusc_map = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
#fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=2)
#fig.show()


#nuscenes_loader.plotMasks(agents_list[0])


