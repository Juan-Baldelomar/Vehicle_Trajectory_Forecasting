"""This file contains code that models the dataset general structure in which a specific dataset is stored.
   For example, this file contains the implementations of both nuscenes and shifts datasets
"""

import matplotlib.pyplot as plt
from ysdc_dataset_api.utils import get_to_track_frame_transform, read_scene_from_file, VehicleTrack
from ysdc_dataset_api.features import FeatureRenderer
import numpy as np


# --------------------------------------------------------------------------- BASE CLASS ---------------------------------------------------------------------------
class AgentTimestep:
    def __init__(self, x: float, y: float, rot):
        self.x = x
        self.y = y
        self.rot = rot


class Agent:
    def __init__(self, agent_id, ego_id, map_name=None):
        self.ego_id = ego_id
        self.agent_id = agent_id
        self.indexes = []
        self.map_name = map_name
        self.timesteps = {}             # dict of steps in time

    def add_step(self, step_id, step):
        if self.timesteps.get(step_id) is None:
            self.timesteps[step_id] = step

    def init_neighbors(self):
        return {self.agent_id: 1}


class EgoVehicle(Agent):
    def __init__(self, agent_id, ego_id, map_name=None):
        super(EgoVehicle, self).__init__(agent_id, ego_id, map_name)

    def init_neighbors(self):
        return {}


class BitmapFeature:
    def __init__(self):
        pass

    def getMasks(self, timestep: AgentTimestep, map_name, **kwargs):
        raise NotImplementedError


class Context:
    def __init__(self, context_id, location=None):
        self.context_id = context_id
        self.neighbors = {}
        self.non_pred_neighbors = {}
        self.humans = {}
        self.objects = {}
        self.map = location

    def add_pred_neighbor(self, agent_id):
        if self.neighbors.get(agent_id) is None:
            self.neighbors[agent_id] = 1

    def add_non_pred_neighbor(self, agent_id):
        if self.non_pred_neighbors.get(agent_id) is None:
            self.non_pred_neighbors[agent_id] = 1


# -------------------------------------------------------------- NUSCENES IMPLEMENTATIONS --------------------------------------------------------------
class Egostep(AgentTimestep):
    def __init__(self, x, y, rot):
        super(Egostep, self).__init__(x, y, rot)


class NuscenesBitmap(BitmapFeature):
    def __init__(self, maps):
        super(NuscenesBitmap, self).__init__()
        self.maps = maps
        self.nusc_ends = {'singapore-onenorth': (1500, 2000),
                          'singapore-queenstown': (3200, 3500),
                          'singapore-hollandvillage': (2700, 3000),
                          'boston-seaport': (3000, 2200)}

    def getMasks(self, timestep: Egostep, map_name, height=200, width=200, canvas_size=(512, 512)):
        """
        function to get the bitmaps of an agent's positions
        :param timestep   : angle of rotation of the masks
        :param map_name   : map name attribute from where to retrieve the map (nuscenes case: string that indicates dict key)
        :param height     : height of the bitmap
        :param width      : width of the bitmap
        :param canvas_size: width of the bitmap
        :return           : list of bitmaps (each mask contains 2 bitmaps)
        """
        if self.maps is None:
            raise ValueError("maps arg should not be None")
        # get map
        nusc_map = self.maps[map_name]
        x, y, yaw = timestep.x, timestep.y, timestep.rot * 180 / np.pi
        # build patch
        patch_box = (x, y, height, width)
        patch_angle = 0  # Default orientation (yaw=0) where North is up
        layer_names = ['drivable_area', 'lane']
        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        return map_mask

    def get_map(self, name, x_start, y_start, x_offset=100, y_offset=100, dpi=25.6):
        """
        function to store map from desired coordinates
        :param name: name of the file to store the map
        :param x_start: x coordinate from which to show map
        :param y_start: y coordinate from which to show map
        :param x_offset: x offset to final, i.e x_final = x_start + x_offset
        :param y_offset: y offset to final, i.e x_final = y_start + y_offset
        :param dpi: resolution of image, example 25.6  gets an image of 256 x 256 pixels
        :return: None
        """
        nusc_map, ends = self.maps[self.map_name], self.nusc_ends[self.map_name]
        x_final = min(ends[0], x_start + x_offset)
        y_final = min(ends[1], y_start + y_offset)
        my_patch = (x_start, y_start, x_final, y_final)
        fig, ax = nusc_map.render_map_patch(my_patch, ['lane', 'lane_divider', 'road_divider', 'drivable_area'], \
                                            figsize=(10, 10), render_egoposes_range=False, render_legend=False, alpha=0.55)
        fig.savefig(name, format="png", dpi=dpi)
        plt.close(fig)


class NuscenesAgentTimestep(Egostep):
    def __init__(self, x: float, y: float, rot, speed: float, accel: float,
                 heading_rate: float, ego_pos_x: float, ego_pos_y: float, ego_rot):

        super(NuscenesAgentTimestep, self).__init__(x, y, rot)
        self.speed = speed
        self.accel = accel
        self.heading_rate = heading_rate
        self.ego_pos_x = ego_pos_x
        self.ego_pos_y = ego_pos_y
        self.ego_rot = ego_rot


# **************************************  non ego vehicle agent **************************************
class NuscenesEgoVehicle(EgoVehicle):
    def __init__(self, ego_id, map_name=None):
        super(NuscenesEgoVehicle, self).__init__(ego_id, ego_id, map_name)

    def get_features(self, timestep_id, origin_timestep=None):
        x_o, y_o, origin_rot = 0, 0, 0
        if origin_timestep is not None:
            x_o = origin_timestep.x
            y_o = origin_timestep.y
            origin_rot = origin_timestep.rot

        agent_time_step = self.timesteps[timestep_id]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = agent_time_step.rot - origin_rot
        return x_pos, y_pos, rel_rot, vel, acc


class NuscenesAgent(Agent):
    def __init__(self, agent_id, ego_id, map_name):
        super(NuscenesAgent, self).__init__(agent_id, ego_id, map_name)
        self.scene_token = None

    def add_step(self, step_id: str, agent_step: NuscenesAgentTimestep):
        self.timesteps[step_id] = agent_step

    def get_features(self, timestep_id, origin_timestep=None):
        x_o, y_o, origin_rot = 0, 0, 0
        if origin_timestep is not None:
            x_o = origin_timestep.x
            y_o = origin_timestep.y
            origin_rot = origin_timestep.rot

        agent_time_step = self.timesteps[timestep_id]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = agent_time_step.rot - origin_rot
        return x_pos, y_pos, rel_rot, vel, acc


# -------------------------------------------------------------- SHIFTS IMPLEMENTATIONS --------------------------------------------------------------
class ShiftsEgoStep(AgentTimestep):
    def __init__(self, x: float, y: float, rot: float, x_speed: float, y_speed: float, x_accel: float,
                 y_accel: float):
        super(ShiftsEgoStep, self).__init__(x, y, rot)
        self.x_speed = x_speed
        self.x_accel = x_accel
        self.y_speed = y_speed
        self.y_accel = y_accel
        self.speed = np.sqrt(x_speed ** 2 + y_speed ** 2)
        self.accel = np.sqrt(x_accel ** 2 + y_accel ** 2)


class ShiftTimeStep(ShiftsEgoStep):
    def __init__(self, x: float, y: float, rot: float, x_speed: float, y_speed: float, x_accel: float,
                 y_accel: float, ego_pos_x: float, ego_pos_y: float, ego_rot: float):
        super(ShiftTimeStep, self).__init__(x, y, rot, x_speed, y_speed, x_accel, y_accel)
        self.ego_pos_x = ego_pos_x
        self.ego_pos_y = ego_pos_y
        self.ego_rot = ego_rot


class ShiftsBitmap(BitmapFeature):
    def __init__(self, renderer_config=None, rows=256, cols=256, resolution=1):
        super(ShiftsBitmap, self).__init__()
        self.renderer = None
        self.set_renderer(renderer_config, rows, cols, resolution)

    def getMasks(self, timestep: ShiftTimeStep, map_name, dummy_param=None):
        scene = read_scene_from_file(map_name)
        # track = scene.past_ego_track[0]
        # create virtual Track
        track = VehicleTrack()
        track.position.x = timestep.x
        track.position.y = timestep.y
        track.yaw = timestep.rot
        track.linear_velocity.x = timestep.x_speed
        track.linear_velocity.y = timestep.y_speed
        track.linear_acceleration.x = timestep.x_accel
        track.linear_acceleration.y = timestep.y_accel
        # transform
        to_track_frame_tf = get_to_track_frame_transform(track)
        feature_maps = self.renderer.produce_features(scene, to_track_frame_tf)['feature_maps']
        virtual_img = np.concatenate([feature_maps[4][np.newaxis, :, :], (feature_maps[7] * 0.5)[np.newaxis, :, :]],
                                     axis=0)
        return virtual_img

    def set_renderer(self, renderer_config=None, rows=256, cols=256, resolution=1):
        if renderer_config is None:
            # Define a renderer config
            renderer_config = {
                # parameters of feature maps to render
                'feature_map_params': {
                    'rows': rows,
                    'cols': cols,
                    'resolution': resolution,  # number of meters in one pixel
                },
                'renderers_groups': [
                    {
                        'time_grid_params': {
                            'start': 24,
                            'stop': 24,
                            'step': 1,
                        },
                        'renderers': [
                            {
                                'road_graph': [
                                    'crosswalk_occupancy',
                                    'crosswalk_availability',
                                    'lane_availability',
                                    'lane_direction',
                                    'lane_occupancy',
                                    'lane_priority',
                                    'lane_speed_limit',
                                    'road_polygons',
                                ]
                            }
                        ]
                    }
                ]
            }
            self.renderer = FeatureRenderer(renderer_config)


class ShiftsEgoVehicle(EgoVehicle):
    def __init__(self, agent_id, map_name=None):
        super(ShiftsEgoVehicle, self).__init__(agent_id, agent_id, map_name)

    def get_features(self, timestep_id, origin_timestep=None):
        x_o, y_o, origin_rot = 0, 0, 0
        if origin_timestep is not None:
            x_o = origin_timestep.x
            y_o = origin_timestep.y
            origin_rot = origin_timestep.rot

        agent_time_step = self.timesteps[timestep_id]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = agent_time_step.rot - origin_rot
        return x_pos, y_pos, rel_rot, vel, acc


class ShiftsAgent(Agent):
    def __init__(self, agent_id, ego_id, map_name=None):
        super(ShiftsAgent, self).__init__(agent_id, ego_id, map_name)

    def get_features(self, timestep_id, origin_timestep=None):
        x_o, y_o, origin_rot = 0, 0, 0
        if origin_timestep is not None:
            x_o = origin_timestep.x
            y_o = origin_timestep.y
            origin_rot = origin_timestep.rot

        agent_time_step = self.timesteps[timestep_id]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = agent_time_step.rot - origin_rot
        return x_pos, y_pos, rel_rot, vel, acc


# -------------------------------------------------------------- DATASET CLASS --------------------------------------------------------------
class Dataset:
    """
    class to model the dataset.
    self.agents is a dictionary that stores agent information with agent_id as key and agent object (implement own agent class
    as needed) as value.

    self.contexts is a dictionary that stores all the timesteps  (sample in nuscenes context) and their information as neighbors
    that appear in a scene. So if you want to get all the neighbors from a timestep, the information is stored in this dictionary
    with a Context object.

    self.ego_vehicles is a dictionary similar to the agents (implement own ego vehicle class as needed), but with ego_vehicles
    information.

    self.non_pred_agents is a dictionary similar to self.agents, but these are agents that are not candidates to a trajectory
    prediction (depends on the dataset to determine the ones that are and the ones that are not candidates)
    """
    def __init__(self, verbose=True):
        self.agents = {}
        self.non_pred_agents = {}
        self.contexts: {str: Context} = {}
        self.ego_vehicles: dict[str] = {}
        self.verbose = verbose

    def add_agent(self, agent_id, agent):
        if self.agents.get(agent_id) is None:
            self.agents[agent_id] = agent
        else:
            print("[WARN]: trying to add agent again")

    def add_context(self, context_id, context):
        if self.contexts.get(context_id) is None:
            self.contexts[context_id] = context

    def add_ego_vehicle(self, ego_id, egovehicle):
        if self.ego_vehicles.get(ego_id) is None:
            self.ego_vehicles[ego_id] = egovehicle

    def insert_context_neighbor(self, agent_id: str, context_id: str):
        self.contexts[context_id].add_pred_neighbor(agent_id)

    # get index of start and end of a trajectory for the ego vehicle
    def get_trajectories_indexes(self, use_ego_vehicles=True, L=-1, overlap=0, min_neighbors=0):
        """
        get trajectories indexes (multiple trajectories can be obtained from a scene). This function gets the start and end
        indexes of each subtrajectory of the scene trajectory
        :param use_ego_vehicles: use real ego_vehicles if True, else use each agent of self.agents as ego vehicle.
        :param L               : lenght of the trajectory. If -1, use all the points in trajectory
        :param overlap         : number of overlap points between subtrajectories of the main trajectory. 0 means no overlap,
                                 so if a trajectory contains 40 points, and L=20, indexes are going to be [(0, 20), (20, 40)]
        :param min_neighbors   : minimum number of neighbors an ego-vehicle starting point needs to have to be considered.
        :return                : None
        """
        ego_vehicles: dict = self.ego_vehicles if use_ego_vehicles else self.agents
        for ego_id, ego_vehicle in ego_vehicles.items():
            timesteps = list(ego_vehicle.timesteps.keys())
            if L == -1:
                ego_vehicle.indexes.append((0, len(timesteps)))
            else:
                start, end = 0, L
                while end <= len(timesteps):
                    # verify if start has enough neighbors as requested
                    if len(self.contexts[timesteps[start]].neighbors) < min_neighbors:
                        start += 1
                        end += 1
                        continue
                    # verify if trajectory can be built
                    if end <= len(timesteps):
                        ego_vehicle.indexes.append((start, end))
                        start = end - overlap
                        end = start + L

    def get_agent_neighbors(self, agent: Agent, kth_traj):
        if len(agent.indexes) == 0:
            return None
        # get start and end of trajectory
        start, end = agent.indexes[kth_traj]
        # get timestep keys
        timestep_keys = list(agent.timesteps.keys())
        # first neighbor is always the agent itself
        neighbors = agent.init_neighbors()
        pos_available = len(neighbors) + 1              # +1 to account for the fact that ego-vehicle occupies position 0

        # traverse all contexts (sample_annotation)
        for key in timestep_keys[start: end]:
            # traverse all neighbors and add the ones that are not yet
            for neighbor_id in self.contexts[key].neighbors:
                if neighbors.get(neighbor_id) is None:
                    neighbors[neighbor_id] = pos_available
                    pos_available += 1

        return neighbors

    def get_prediction_agents(self, size, skip=0, mode='single', overlap_points=0):
        """
        Sometimes datasets can be really heavy and even require transformations to the data that might result in a lot of resources
        as time and memory invested. In that case it could be helpful to retrieve the agents keys for which you want to perform a prediction
        in training or during inference. You could then pass this agents keys and the dictionary containing all the dataset information
        and retrieve the information with the transformations needed with a tensorflow or pytorch pipeline. As the information is stored
        in a dictionary, retrieving the information needed in the pipeline could be achieved in constant time.

        :param size indicates the minimum size of points in the trajectory (see get_trajectories_indexes doc)
        :param skip indicates points to skip from agents data retrived (see get_trajectories_indexes doc)
        :param mode indicates how to treat trajectories that are bigger than the minimum size (see get_trajectories_indexes doc)
        :param overlap_points indicates the number of points that two consecutive sub-trajectories can have
        :return: List of agent keys
        """
        self.get_trajectories_indexes(size, skip, mode, overlap_points)
        agent_keys = []

        for (key, agent) in self.agents.items():
            if len(agent.indexes > 0):
                agent_keys.append(key)

        return agent_keys

