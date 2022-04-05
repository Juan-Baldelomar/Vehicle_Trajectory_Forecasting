
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
            # join past and future steps
            steps = list(scene.past_vehicle_tracks) + list(scene.future_vehicle_tracks)
            ego_steps = list(scene.past_ego_tracks) + list(scene.future_ego_tracks)
            # load ego vehicle and contexts of the scene
            self.load_ego_vehicles_and_context(scene.id, ego_steps)

            # traverse each past timestep
            for i, (track_step, ego_step) in enumerate(zip(steps, ego_steps)):
                # traverse all agents
                for track in track_step.tracks:
                    # ignore those that are not candidates to be used to predict
                    if track.track_id not in prediction_requests_ids:
                        continue

                    # build a unique agent id in all dataset
                    agent_id = scene.id + str(track.track_id)
                    # CREATE agent if does not exist
                    if agents.get(agent_id) is None:
                        # verbose mode
                        print('new agent: ', agent_id) if self.verbose else None
                        agents[agent_id] = ShiftsAgent(agent_id, None)
                    # append agent
                    agent = agents[agent_id]
                    append_agent_data(agent, track, ego_step)
                    # insert agent as neighbor (scene.id + i = context_id or same as step_id)
                    self.dataset.insert_context_neighbor(agent_id, scene.id + str(i))

            # FOR track_step, ego in zip(...)
        return agents

    def load_ego_vehicles_and_context(self, ego_id, ego_steps, location=None):
        ego_vehicle = EgoVehicle(ego_id, location)
        self.dataset.add_ego_vehicle(ego_id, ego_vehicle)

        for i, step in enumerate(ego_steps):
            # step_id should be the id of the context object in the Context scene
            step_id = ego_id + '_' + i
            self.dataset.add_context(step_id, Context(step_id))
            ego_vehicle.add_step(step_id, Egostep(step.position.x, step.position.y, step.yaw))
