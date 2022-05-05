
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc

from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import get_file_paths, scenes_generator, transform_2d_points, \
    get_to_track_frame_transform, get_latest_track_state_by_id, read_feature_map_from_file, request_is_valid


# Load protobufs for training dataset
dataset_path = '/data/shifts/data/development'
filepaths = get_file_paths(dataset_path)

scene = next(scenes_generator(filepaths))

past_steps = scene.past_vehicle_tracks
future_steps = scene.future_vehicle_tracks
ego_past_steps = scene.past_ego_track
# counter = 0
# for scene in scenes_generator(filepaths):
#     print(counter)
#     counter += 1

# Number of known history steps
# Index 0 is farthest (-5s) into the past, and index 24 represents current time
print('Number of history steps:', len(scene.past_vehicle_tracks))


print('Number of vehicles seen at current time:', len(scene.past_vehicle_tracks[-1].tracks))
print(scene.past_vehicle_tracks[-1].tracks)

tr = scene.past_vehicle_tracks[-1].tracks

tr[2].track_id in scene.prediction_requests

pr = scene.prediction_requests

pr_ids = [p.track_id for p in pr]

pr_ids



ego_tr = scene.past_ego_track

ego_tr[0]