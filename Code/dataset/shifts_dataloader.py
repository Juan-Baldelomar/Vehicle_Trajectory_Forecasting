
# base class
from Code.dataset.dataloader import Loader

# data manipulation and tools
import numpy as np
from Code.dataset.DataModel import *
import os

# shifts libraries
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import get_file_paths, scenes_generator, transform_2d_points, VehicleTrack


class ShiftsLoader(Loader):
    def __init__(self, DATAROOT, pickle=True, pickle_filename='/data/shifts/data.pkl', chunk=(0, 1000), verbose=True):
        # super constructor
        super(ShiftsLoader, self).__init__(DATAROOT, verbose)
        self.renderer = None
        # flag to indicate if data can be loaded from pickle files
        pickle_ok: bool = os.path.isfile(pickle_filename)

        if pickle and pickle_ok:
            # its okay to read pickle files to load data
            self.load_pickle_data(pickle_filename)
        else:
            # load data from scratch
            self.load_data(chunk)
            self.save_pickle_data(pickle_filename)

        ShiftsAgent.context_dict = self.dataset.contexts

    def load_ego_vehicles_and_context(self, ego_id, ego_steps, location=None):
        ego_vehicle = ShiftsEgoVehicle(ego_id, location)
        self.dataset.add_ego_vehicle(ego_id, ego_vehicle)

        for i, step in enumerate(ego_steps):
            # step_id should be the id of the context object in the Context scene
            step_id = ego_id + '_' + str(i)
            self.dataset.add_context(step_id, Context(step_id))
            step = ShiftsEgoStep(step.position.x, step.position.y, step.yaw, step.linear_velocity.x,
                                 step.linear_velocity.y, step.linear_acceleration.x, step.linear_acceleration.y)
            ego_vehicle.add_step(step_id, step)

    def load_data(self, chunk=(0, 1000)):
        def get_step(track, ego):
            agent_step = ShiftTimeStep(track.position.x, track.position.y, track.yaw, track.linear_velocity.x,
                                       track.linear_velocity.y, track.linear_acceleration.x, track.linear_acceleration.y,
                                       ego.position.x, ego.position.y, ego.yaw)
            return agent_step

        filepaths = get_file_paths(self.DATAROOT)
        scenes_paths = scenes_generator(filepaths[chunk[0]: chunk[1]], yield_fpath=True)
        # traverse scenes
        for scene, path in scenes_paths:
            #get tracks of interest
            prediction_requests_ids = {pr.track_id for pr in scene.prediction_requests}
            # join past and future steps
            timesteps = list(scene.past_vehicle_tracks) + list(scene.future_vehicle_tracks)
            ego_steps = list(scene.past_ego_track) + list(scene.future_ego_track)
            # load ego vehicle and contexts of the scene
            self.load_ego_vehicles_and_context(scene.id, ego_steps, location=path)

            # traverse each past timestep
            for i, (track_step, ego_step) in enumerate(zip(timesteps, ego_steps)):
                context_id = scene.id + '_' + str(i)
                # traverse all agents
                for track in track_step.tracks:
                    # build a unique agent id in all dataset
                    agent_id = scene.id + '_' + str(track.track_id)
                    # if agent IS NOT A CANDIDATE FOR PREDICTION, add as non prediction agent
                    if track.track_id not in prediction_requests_ids:
                        # CREATE agent if does not exist. Use path as map name (SEE ShitAgent doc)
                        if self.dataset.non_pred_agents.get(agent_id) is None:
                            self.dataset.non_pred_agents[agent_id] = ShiftsAgent(agent_id, scene.id, path)
                        # insert timestep, step_id = context_id
                        self.dataset.non_pred_agents[agent_id].add_step(context_id, get_step(track, ego_step))
                        # insert as non prediction neighbor
                        self.dataset.contexts[context_id].add_non_pred_neighbor(agent_id)
                    else:
                        # CREATE agent if does not exist. Use path as map name (SEE ShitAgent doc)
                        if self.dataset.agents.get(agent_id) is None:
                            print('new agent: ', agent_id) if self.verbose else None
                            self.dataset.agents[agent_id] = ShiftsAgent(agent_id, scene.id, path)
                        # insert timestep, step_id = context_id
                        self.dataset.agents[agent_id].add_step(context_id, get_step(track, ego_step))
                        # insert agent as neighbor (scene.id + i = context_id or same as step_id)
                        self.dataset.contexts[context_id].add_pred_neighbor(agent_id)
