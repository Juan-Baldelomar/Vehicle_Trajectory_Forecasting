import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

nusc_ends = {'singapore-onenorth': (1500, 2000),
             'singapore-queenstown': (3200, 3500),
             'singapore-hollandvillage': (2700, 3000),
             'boston-seaport': (3000, 2200)}


# --------------------------------------------------------------------------- BASE CLASS ---------------------------------------------------------------------------
class Egostep:
    def __init__(self, x, y, rot):
        self.x = x
        self.y = y
        self.rot = rot


class Context:
    def __init__(self, context_id, location=None):
        self.context_id = context_id
        self.neighbors = set()
        self.non_pred_neighbors = {}
        self.humans = {}
        self.objects = {}
        self.map = location

    def add_non_pred_neighbor(self, agent_id, agent_timestep):
        if self.non_pred_neighbors.get(agent_id) is None:
            self.non_pred_neighbors[agent_id] = agent_timestep


class EgoVehicle:
    def __init__(self, ego_id, map_name=None):
        self.agent_id = ego_id
        self.indexes = []
        self.map_name = map_name
        self.timesteps = {}             # dict of steps in time

    def add_step(self, step_id, ego_step):
        if self.timesteps.get(step_id) is None:
            self.timesteps[step_id] = ego_step

    def get_neighbors(self, context_pool):
        """
        :param context_pool: dictionary of Context objects
        :return:
        """
        neighbors_across_time = set()
        for timestep_id, ego_step in self.timesteps.items():
            for neighbor in context_pool[timestep_id].neighbors:
                neighbors_across_time.add(neighbor)

        return neighbors_across_time

    def getMasks(self, maps: dict, timestep: Egostep, path: str, name: str, height=200, width=200, canvas_size=(512, 512)):
        """
        function to get the bitmaps of an agent's positions
        :param maps: maps dictionary
        :param timestep: angle of rotation of the masks
        :param path: angle of rotation of the masks
        :param name: angle of rotation of the masks
        :param height: height of the bitmap
        :param width:  width of the bitmap
        :param canvas_size:  width of the bitmap
        :return: list of bitmaps (each mask contains 2 bitmaps)
        """
        # get map
        nusc_map = maps[self.map_name]
        x, y, rot = timestep.x, timestep.y, Quaternion(timestep.rot)
        yaw = rot.yaw_pitch_roll[0] * 180 / np.pi

        # build patch
        patch_box = (x, y, height, width)
        patch_angle = 0  # Default orientation (yaw=0) where North is up
        layer_names = ['drivable_area', 'lane']
        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
        return map_mask
        # for layer in layer_names:
        #     fig, ax = nusc_map.render_map_mask(patch_box, patch_angle, [layer], canvas_size, figsize=(12, 4), n_row=1)
        #     fig.savefig('/'.join([path, layer, name]), format="png", dpi=canvas_size[0] / 10)
        #     plt.close(fig)

    def get_map(self, maps: dict, name, x_start, y_start, x_offset=100, y_offset=100, dpi=25.6):
        """
        function to store map from desired coordinates
        :param maps: map dictionary that containts the name as key and map object as value
        :param name: name of the file to store the map
        :param x_start: x coordinate from which to show map
        :param y_start: y coordinate from which to show map
        :param x_offset: x offset to final, i.e x_final = x_start + x_offset
        :param y_offset: y offset to final, i.e x_final = y_start + y_offset
        :param dpi: resolution of image, example 25.6  gets an image of 256 x 256 pixels
        :return: None
        """
        nusc_map, ends = maps[self.map_name], nusc_ends[self.map_name]
        x_final = min(ends[0], x_start + x_offset)
        y_final = min(ends[1], y_start + y_offset)
        my_patch = (x_start, y_start, x_final, y_final)
        fig, ax = nusc_map.render_map_patch(my_patch, ['lane', 'lane_divider', 'road_divider', 'drivable_area'], \
                                            figsize=(10, 10), render_egoposes_range=False, render_legend=False, alpha=0.55)
        fig.savefig(name, format="png", dpi=dpi)
        plt.close(fig)


# --------------------------------------------------------------------- NON EGO VEHICLE AGENT ---------------------------------------------------------------------
class AgentTimestep(Egostep):
    def __init__(self, x: float, y: float, rot, speed: float, accel: float,
                 heading_rate: float, ego_pos_x: float, ego_pos_y: float, ego_rot):

        super(AgentTimestep, self).__init__(x, y, rot)
        self.speed = speed
        self.accel = accel
        self.heading_rate = heading_rate
        self.ego_pos_x = ego_pos_x
        self.ego_pos_y = ego_pos_y
        self.ego_rot = ego_rot


class Agent(EgoVehicle):
    context_dict = None

    def __init__(self, agent_id, map_name):
        super(Agent, self).__init__(agent_id, map_name)
        self.scene_token = None

    def add_step(self, step_id: str, agent_step: AgentTimestep):
        self.timesteps[step_id] = agent_step

    def get_map_patch(self, x_start, y_start, x_offset=100, y_offset=100):
        ends = nusc_ends[self.map_name]
        x_final = min(ends[0], x_start + x_offset)
        y_final = min(ends[1], y_start + y_offset)
        return x_start, y_start, x_final, y_final

    # return list of unique neighbors through all the trajectory
    def get_neighbors(self, kth_traj):
        if len(self.indexes) == 0:
            return None

        start, end = self.indexes[kth_traj]
        keys = list(self.timesteps.keys())
        neighbors = {}
        pos_available = 0

        # traverse all contexts (sample_annotation)
        for key in keys[start: end]:
            context = Agent.context_dict[key]

            # traverse all neighbors and add the ones that are not yet
            for neighbor_id in context.neighbors:
                if neighbors.get(neighbor_id) is None:
                    neighbors[neighbor_id] = pos_available
                    pos_available += 1

        return neighbors

    def get_features(self, timestep_id, origin_timestep=None, use_ego=True):
        x_o, y_o, origin_rot = 0, 0, (0, 0, 0, 1)
        if origin_timestep is not None:
            if use_ego:
                x_o = origin_timestep.x
                y_o = origin_timestep.y
                origin_rot = origin_timestep.rot
            else:
                x_o = origin_timestep.x
                y_o = origin_timestep.y
                origin_rot = origin_timestep.rot

        agent_time_step = self.timesteps[timestep_id]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = Quaternion(origin_rot).inverse * Quaternion(agent_time_step.rot)
        yaw, _, _ = rel_rot.yaw_pitch_roll
        yaw = yaw  # in radians
        return x_pos, y_pos, yaw, vel, acc


class Dataset:
    """
    class to model the dataset.
    self.agents is a dictionary that stores agent information with agent_id as key and agent object as value
    self.contexts is a dictionary that stores all the timesteps  (sample in nuscenes context) and their information as neighbors that appear in a scene.
        so if you want to get all the neighbors from a timestep, the information is stored in this dictionary with a Context object.
    self.ego_vehicles is a dictionary similar to the agents, but with ego_vehicles information.
    """
    def __init__(self):
        self.agents = {}
        self.contexts: dict[Context] = {}
        self.ego_vehicles: dict[EgoVehicle] = {}

    def add_agent(self, agent_id, agent):
        if self.agents.get(agent_id) is None:
            self.agents[agent_id] = agent
        else:
            print("[WARN]: trying to add agent again")

    def add_context(self, context_id):
        if self.contexts.get(context_id) is None:
            self.contexts[context_id] = Context(context_id)

    def add_ego_vehicle(self, ego_id, egovehicle):
        if self.ego_vehicles.get(ego_id) is None:
            self.ego_vehicles[ego_id] = egovehicle
