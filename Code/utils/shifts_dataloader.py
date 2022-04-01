
# base class
from dataloader import Loader
from dataloader import save_pkl_data

# data manipulation and tools
import numpy as np
from Dataset import *
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

    def load_data(self, chunk=(0, 1000)):
        def append_agent_data(agent, track, ego, step_id):
            speed = np.sqrt(track.linear_velocity.x**2 + track.linear_velocity.y**2)
            accel = np.sqrt(track.linear_acceleration.x ** 2 + track.linear_acceleration.y ** 2)
            agent_step = AgentTimestep(track.position.x, track.position.y, track.yaw, speed,
                                       accel, 0, ego.position.x, ego.position.y, ego.yaw)
            agent.add_step(step_id, agent_step)

        filepaths = get_file_paths(self.DATAROOT)
        scenes = scenes_generator(filepaths[chunk[0]: chunk[1]])
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
                        agents[agent_id] = Agent(agent_id, None)

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

    def load_ego_vehicles(self, ego_id, ego_steps, location=None):
        ego_vehicle = EgoVehicle(ego_id, location)
        self.dataset.add_ego_vehicle(ego_id, ego_vehicle)

        for i, step in enumerate(ego_steps):
            # step_id should be the id of the context object in the Context scene
            step_id = ego_id + '_' + i
            ego_vehicle.add_step(step_id, Egostep(step.position.x, step.position.y, step.yaw))

    def get_context_information(self):
        pass

# shifts_data = ShiftsLoader('/data/shifts/data/development', pickle=False)
#
# trajs = shifts_data.get_trajectories_data(size=50, offset=25)
#
# trajs['abs_pos'] = np.array(trajs['abs_pos'])
# trajs['rel_pos'] = np.array(trajs['rel_pos'])
# trajs['rotation'] = np.array(trajs['rotation'])
# trajs['rel_rotation'] = np.array(trajs['rel_rotation'])
# trajs['speed'] = np.array(trajs['speed'])
# trajs['accel'] = np.array(trajs['accel'])
#
# trajs['abs_pos'][0]
#
# save_pkl_data(trajs, '/data/shifts/sample_shifts_data.pkl')
#
# en = list(shifts_data.dataset['agents'].keys())
# len(en)