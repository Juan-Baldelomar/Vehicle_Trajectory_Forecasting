import glob

import numpy as np
import os
from Code.dataset.nuscenes_dataloader import NuscenesLoader
from Code.dataset.shifts_dataloader import ShiftsLoader
from Code.dataset.InputQuery import *
from Code.utils import save_utils as dl


# def data_statistics(data_start, data_end):
#     assert (data_start >= 1 and data_end <= 18)
#     dataroot = '/data/shifts/data/development'
#
#     for chunk_number in range(data_start, data_end + 1):
#         # path of preloaded data
#         pickle_filename = '/data/shifts/data_chunk' + str(chunk_number) + '.pkl'
#         chunk_start = (chunk_number - 1) * 2000
#         chunk_end = chunk_number * 2000
#         shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=pickle, pickle_filename=pickle_filename,
#                                      chunk=(chunk_start, chunk_end))


def shifts_extraction(past_length, future_length, neighbors, get_bitmaps: bool, origin_offset=None,
                      pickle=True, data_start=1, data_end=18, force_overwrite=False):
    assert(data_start >= 1 and data_end <= 188)
    dataroot = '/data/shifts/data/train'

    if origin_offset is None:
        origin_offset = past_length - 1

    for chunk_number in range(data_start, data_end + 1):
        # path of preloaded data
        pickle_filename = '/data/shifts/train/data_chunk' + str(chunk_number) + '.pkl'
        # path to store inputs
        maps_path  = '../data/shifts/train/neigh_' + str(neighbors) + '/maps/chunk' + str(chunk_number)
        final_path = '../data/shifts/train/neigh_' + str(neighbors) + '/shifts_data_chunk' + str(chunk_number) + '.pkl'
        # create needed directories
        dl.valid_path(maps_path, os.path.dirname(final_path))
        # START EXTRACTION
        chunk_start = (chunk_number-1) * 2000
        chunk_end = chunk_number * 2000
        shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=pickle, pickle_filename=pickle_filename, chunk=(chunk_start, chunk_end))
        inputQuery = InputQuery(shifts_loader)
        shifts_bitmap = ShiftsBitmap() if get_bitmaps else None
        inputs = inputQuery.get_TransformerCube_Input(past_length, future_length, neighbors, origin_offset,
                                                      bitmap_extractor=shifts_bitmap, path=maps_path)

        # dealing with existing files
        if os.path.isfile(final_path) and not force_overwrite:
            print('file: ', final_path, ' exists. Do you want to overwrite it?:')
            answer = input()
            if answer.lower() == 'yes':
                dl.save_pkl_data(inputs, final_path)
            else:
                print('[WARN] IGNORING DATA')
        else:
            dl.save_pkl_data(inputs, final_path)


def join_data(prepath, start, end):
    assert(start >= 1 and end <= 188)
    inputs = []
    for i in range(start, end+1):
        print('joining file:', i)
        path = prepath + str(i) + '.pkl'
        input_ = dl.load_pkl_data(path)
        inputs += input_

    dl.save_pkl_data(inputs, '../data/shifts/train/neigh_5/shifts_data_all_p2_v4.pkl', protocol=4)


def nuscenes_extraction(past_length, future_length, neighbors, get_bitmaps: bool, origin_offset=None,
                        pickle=True, data_partition='train'):
    # PATH
    dataroot_base = '/data/sets/nuscenes'
    dataroot_train = '/media/juan/Elements'
    # dataset attributes
    dataroot = dataroot_train + dataroot_base
    # VERSION
    version = 'v1.0-trainval'

    # NAME
    data_name = data_partition
    if origin_offset is None:
        origin_offset = past_length - 1

    # path to store inputs
    maps_path  = '../data/nuscenes/maps'
    final_path = '../data/nuscenes/train/neigh_' + str(neighbors) + '/nuscenes_extra_data.pkl'
    # create needed directories
    dl.valid_path(maps_path, os.path.dirname(final_path))
    # START EXTRACTION
    nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=pickle, version=version, data_name=data_name, loadMap=True)
    inputQuery = InputQuery(nuscenes_loader)
    nusc_bitmap = NuscenesBitmap(nuscenes_loader.maps) if get_bitmaps else None
    inputs = inputQuery.get_TransformerCube_Input(past_length, future_length, neighbors, origin_offset,
                                                  use_ego_vehicles=False, bitmap_extractor=nusc_bitmap, path=maps_path)
    dl.save_pkl_data(inputs, final_path)


def get_origin(dataroot):
    origins_info = {}
    files = glob.glob(dataroot)
    for filename in files:
        print('processing file: ', filename)
        data:Dataset = dl.load_pkl_data(filename)
        for ego_vehicle in data.ego_vehicles.values():
            origin = list(ego_vehicle.timesteps.values())[24]
            origins_info[ego_vehicle.agent_id] = [origin.to_tuple(), ego_vehicle.map_name]

    dl.save_pkl_data(origins_info, 'origins_info.pkl', 4)


#data = dl.load_pkl_data('origins_info.pkl')
get_origin('/data/shifts/val/*')
#shifts_extraction(past_length=25, future_length=25, neighbors=5, get_bitmaps=True, pickle=False, data_start=83, data_end=83)
#join_data('../data/shifts/train/neigh_5/shifts_data_chunk', start=95, end=188)
#data1 = dl.load_pkl_data('../data/shifts/train/neigh_5/shifts_data_all_p1_v4.pkl')
#data2 = dl.load_pkl_data('../data/shifts/train/neigh_5/shifts_data_all_p2_v4.pkl')
#final_data = data1 + data2
#dl.save_pkl_data(final_data, '../data/shifts/train/neigh_5/shifts_data_all_p4.pkl')


# nuscenes_extraction(past_length=10, future_length=12, neighbors=5, get_bitmaps=True, pickle=True)
#
#
# data = dl.load_pkl_data('../data/nuscenes/train/neigh_5/nuscenes_data.pkl')
# extra_data = dl.load_pkl_data('../data/nuscenes/train/neigh_5/nuscenes_extra_data.pkl')
# data = data + extra_data
# n = len(data)
# val_len = n//10
#
# indexes = np.arange(n)
# val_index = np.random.choice(n, val_len, replace=False)
# train_index = np.setdiff1d(indexes, val_index)
# val_data = [data[i] for i in val_index]
# train_data = [data[i] for i in train_index]
#
# dl.save_pkl_data(train_data, '../data/nuscenes/train/neigh_5/nuscenes_train.pkl', protocol=4)
# dl.save_pkl_data(val_data, '../data/nuscenes/train/neigh_5/nuscenes_val.pkl', protocol=4)
