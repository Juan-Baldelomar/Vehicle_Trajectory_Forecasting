
from nuscenes_dataloader import NuscenesLoader
from InputQuery import InputQuery


#
dataroot_base = '/data/sets/nuscenes'
# dataroot_train = '/media/juan/Elements'
#
# # dataset attributes
# #dataroot = dataroot_train + dataroot_base
dataroot = dataroot_base
#
# #version = 'v1.0-trainval'
version = 'v1.0-mini'
#
# #data_name = 'train'
data_name = 'mini_train'
#
nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name)

inputQuery = InputQuery(nuscenes_loader)

cubes = inputQuery.get_TransformerCube_Input(N=5)

cubes[5]
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

