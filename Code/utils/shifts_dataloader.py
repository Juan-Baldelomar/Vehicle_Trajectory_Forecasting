
# base class
from dataloader import Loader
from dataloader import save_pkl_data

# data manipulation and tools
import numpy as np
import pickle
import os

# shifts libraries
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import get_file_paths, scenes_generator, transform_2d_points


class ShiftsLoader(Loader):

    def __init__(self, DATAROOT, pickle=True, pickle_filename='/data/shifts/data.pkl', verbose=True):

        # super constructor
        super(ShiftsLoader, self).__init__(DATAROOT, verbose)

        # flag to indicate if data can be loaded from pickle files
        pickle_ok: bool = os.path.isfile(pickle_filename)

        if pickle and pickle_ok:
            # its okay to read pickle files to load data
            self.load_pickle_data(pickle_filename)

        else:
            # load data from scratch
            self.dataset['context'] = {}
            self.dataset['agents'] = self.load_data()
            #self.get_context_information()
            self.save_pickle_data(pickle_filename)

    def load_data(self):

        def append_agent_data(agent: dict, track, ego):
            agent['abs_pos'].append([track.position.x, track.position.y])
            agent['ego_pos'].append([ego.position.x, ego.position.y])
            agent['rotation'].append(track.yaw)
            agent['ego_rotation'].append(ego.yaw)
            agent['speed'].append([track.linear_velocity.x, track.linear_velocity.y])
            agent['accel'].append([track.linear_acceleration.x, track.linear_acceleration.y])

        filepaths = get_file_paths(self.DATAROOT)
        scenes = scenes_generator(filepaths[0:1000])
        agents = {}
        # traverse scenes
        for scene in scenes:

            #get tracks of interest
            prediction_requests_ids = {pr.track_id for pr in scene.prediction_requests}

            # traverse each past timestep
            for track_step, ego in zip(scene.past_vehicle_tracks, scene.past_ego_track):

                # traverse all agents
                for track in track_step.tracks:

                    # ignore those that are not candidates to be used to predict
                    if track.track_id not in prediction_requests_ids:
                        continue

                    agent_id = scene.id + str(track.track_id)           # build a unique agent id in all dataset

                    # CREATE agent if does not exist
                    if agents.get(agent_id) is None:

                        # verbose mode
                        print('new agent: ', agent_id) if self.verbose else None

                        agents[agent_id] = {'abs_pos':        [],                # list of coordinates in the world frame (2d) (sequence)
                                            'ego_pos':        [],                # position of the ego vehicle, the one with the cameras
                                            'rotation':       [],                # list of rotation parametrized as a quaternion
                                            'ego_rotation':   [],                # list of rotations of ego vehicle
                                            'speed':          [],                # list of scalar
                                            'accel':          [],                # list scalar
                                            'context':        [],                # list of ids
                                            }

                    agent = agents[agent_id]
                    append_agent_data(agent, track, ego)

            # FOR track_step, ego in zip(...)

            # traverse each future timestep
            for track_step, ego in zip(scene.future_vehicle_tracks, scene.future_ego_track):
                for track in track_step.tracks:

                    # ignore those that are not candidates to be used to predict
                    if track.track_id not in prediction_requests_ids:
                        continue

                    agent_id = scene.id + str(track.track_id)           # build a unique agent id in all dataset
                    agent = agents[agent_id]
                    append_agent_data(agent, track, ego)

        return agents

    def check_consistency(self):
        """check that the data was load propperly and it fulfills the desired characteristics"""
        raise NotImplementedError

    def _get_custom_trajectories_data(self, **kwargs):
        """
        get absolute positons, rotations, relative positions and relative rotations to a given point in the ego_positions
        and a given rotation of the  trajectories

        :param offset indicates which point of the ego positions will be taken as the origin
        :return dictionary containing the desired values as numpy arrays
        """
        offset = kwargs['offset']
        self.origin_offset = offset

        key_agents = self.dataset['agents'].items()
        trajectories = {'id': [], 'abs_pos': [], 'rel_pos': [], 'rotation': [], 'rel_rotation': [], 'speed': [], 'accel': []}

        for (key, agent) in key_agents:
            indexes = agent['index']
            for index in indexes:
                start, end = index

                # get data to process
                abs_pos = np.array(agent['abs_pos'][start: end])
                rotation = np.array(agent['rotation'][start: end])
                ego_pos = np.array(agent['ego_pos'][start: end])
                ego_rotation = np.array(agent['ego_rotation'][start: end])
                speed = np.array(agent['speed'][start: end])
                accel = np.array(agent['accel'][start: end])

                # get relative positions
                origin = ego_pos[offset]
                rel_pos = abs_pos - origin

                # get relative rotations
                rel_rotation = ego_rotation - rotation

                # append values
                trajectories['id'].append(key)
                trajectories['abs_pos'].append(abs_pos)
                trajectories['rotation'].append(rotation)
                trajectories['rel_pos'].append(rel_pos)
                trajectories['rel_rotation'].append(rel_rotation)
                trajectories['speed'].append(speed)
                trajectories['accel'].append(accel)

        return trajectories


shifts_data = ShiftsLoader('/data/shifts/data/development', pickle=False)

trajs = shifts_data.get_trajectories_data(size=50, offset=25)

trajs['abs_pos'] = np.array(trajs['abs_pos'])
trajs['rel_pos'] = np.array(trajs['rel_pos'])
trajs['rotation'] = np.array(trajs['rotation'])
trajs['rel_rotation'] = np.array(trajs['rel_rotation'])
trajs['speed'] = np.array(trajs['speed'])
trajs['accel'] = np.array(trajs['accel'])

trajs['abs_pos'][0]

save_pkl_data(trajs, '/data/shifts/sample_shifts_data.pkl')

en = list(shifts_data.dataset['agents'].keys())
len(en)