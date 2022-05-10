import numpy as np
import os
from nuscenes_dataloader import NuscenesLoader
from shifts_dataloader import ShiftsLoader
from InputQuery import *
from Code.utils import save_utils as dl


def shifts_extraction(past_length, future_length, neighbors, get_bitmaps: bool, origin_offset=None,
                      pickle=True, data_start=1, data_end=18, force_overwrite=False):
    assert(data_start >= 1 and data_end <= 18)
    dataroot = '/data/shifts/data/development'

    if origin_offset is None:
        origin_offset = past_length - 1

    for chunk_number in range(data_start, data_end + 1):
        pickle_filename = '/data/shifts/data_chunk' + str(chunk_number) + '.pkl'
        maps_path = '../data/maps/shifts/chunk' + str(chunk_number)
        final_path = '../data/shifts_data_chunk' + str(chunk_number) + '.pkl'
        dl.valid_path(maps_path)
        chunk_start = (chunk_number-1) * 2000
        chunk_end = chunk_number * 2000
        shifts_loader = ShiftsLoader(DATAROOT=dataroot, pickle=pickle, pickle_filename=pickle_filename, chunk=(chunk_start, chunk_end))
        inputQuery = InputQuery(shifts_loader)
        shifts_bitmap = ShiftsBitmap() if get_bitmaps else None
        inputs = inputQuery.get_TransformerCube_Input(past_length, future_length, neighbors, origin_offset,
                                                      bitmap_extractor=shifts_bitmap, path=maps_path)

        # dealing with exiting files
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
    assert(start >= 1 and end <= 18)
    inputs = []
    for i in range(start, end+1):
        path = prepath + str(i) + '.pkl'
        input_ = dl.load_pkl_data(path)
        inputs += input_

    dl.save_pkl_data(inputs, '../data/shifts_data_all_p4.pkl', protocol=4)


shifts_extraction(past_length=25, future_length=25, neighbors=5, get_bitmaps=True, pickle=False, data_start=14, data_end=18)
