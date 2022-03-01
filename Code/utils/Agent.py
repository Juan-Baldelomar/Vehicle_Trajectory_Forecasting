
from Dataset import AgentTimestep
import numpy as np
from pyquaternion import Quaternion


class Agent:
    context_dict = None

    def __init__(self, agent_id=None):
        self.agent_id = agent_id
        self.map_name = None
        self.scene_token = None
        self.context = {}           # dictionary of ids of context-scenes (TimeSteps).
        self.index_list = []        # list of indexes that indicate the start and end of the multiple trajectories that can be obtained
                                    # from the same agent

    def add_observation(self, t_context, t_x, t_y, t_rotation, t_speed, t_accel,
                        t_heading_rate, t_ego_pos_x, t_ego_pos_y, t_ego_rotation):

        self.context[t_context] = AgentTimestep(t_x, t_y, t_rotation, t_speed, t_accel,
                                                t_heading_rate, t_ego_pos_x, t_ego_pos_y, t_ego_rotation)

    def plotMasks(self, maps: dict, height=200, width=200):
        """
        exploratory function to plot the bitmaps of an agent's positions
        :param maps: maps dictionary
        :param height: height of the bitmap
        :param width:  width of the bitmap
        :return: None
        """

        # get map
        map = maps[self.map_name]

        # traverse agent positions
        for pos in self.abs_pos:
            x, y = pos[0], pos[1]
            patch_box = (x, y, height, width)
            patch_angle = 0  # Default orientation where North is up
            layer_names = ['drivable_area', 'walkway']
            canvas_size = (1000, 1000)

            figsize = (12, 4)
            fig, ax = map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=figsize, n_row=1)
            fig.show()

    def getMasks(self, maps: dict, yaw=0, height=200, width=200):
        """
         function to get the bitmaps of an agent's positions
        :param maps: maps dictionary
        :param yaw: angle of rotation of the masks
        :param height: height of the bitmap
        :param width:  width of the bitmap
        :return: list of bitmaps (each mask contains 2 bitmaps)
        """

        # get map
        map = maps[self.map_name]

        masks_list = []

        # traverse agents positions
        for pos in self.abs_pos:
            x, y = pos[0], pos[1]
            patch_box = (x, y, height, width)
            patch_angle = yaw  # Default orientation (yaw=0) where North is up
            layer_names = ['drivable_area', 'walkway']
            canvas_size = (1000, 1000)
            map_mask = map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
            masks_list.append(map_mask)

        return masks_list

    # return list of unique neighbors through all the trajectory
    def get_neighbors(self, kth_traj):
        if len(self.index_list) == 0:
            return None

        start, end = self.index_list[kth_traj]
        keys = list(self.context.keys())
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

    def get_transformer_matrix(self, agents: dict, kth_traj: int, offset_origin=-1):
        neighbors_positions = self.get_neighbors(kth_traj)
        start, end = self.index_list[kth_traj]

        traj_size = end - start
        matrix = np.zeros((len(neighbors_positions), traj_size, 2))
        time_steps = list(self.context.keys())

        # use a fixed origin in the agent abs positions
        if offset_origin >= 0:
            x_o, y_o = self.abs_pos[offset_origin]

        for j in range(start, end):

            # use the current abs position of the agent as origin
            if offset_origin < 0:
                x_o, y_o = self.abs_pos[j]

            context_key = time_steps[j]
            agent_neighbor_ids = Agent.context_dict[context_key]['neighbors']

            for neighbor_id in agent_neighbor_ids:
                neighbor: Agent = agents[neighbor_id]
                time_pos = neighbor.context.get(context_key)
                if time_pos is not None:
                    x, y = neighbor.abs_pos[time_pos]
                    i = neighbors_positions[neighbor_id]
                    matrix[i, j, 0] = x - x_o
                    matrix[i, j, 1] = y - y_o

        return matrix

    def get_features(self, timestep, origin_timestep=None, use_ego=True):
        x_o, y_o, origin_rot = 0, 0, (0, 0, 0, 1)
        if origin_timestep is not None:
            if use_ego:
                x_o = self.context[origin_timestep].ego_pos_x
                y_o = self.context[origin_timestep].ego_pos_y
                origin_rot = self.context[origin_timestep].ego_rot
            else:
                x_o = self.context[origin_timestep].x
                y_o = self.context[origin_timestep].y
                origin_rot = self.context[origin_timestep].rot

        agent_time_step = self.context[timestep]
        x_pos = agent_time_step.x - x_o
        y_pos = agent_time_step.y - y_o
        vel = agent_time_step.speed
        acc = agent_time_step.accel
        rel_rot = Quaternion(origin_rot).inverse * Quaternion(agent_time_step.rot)
        yaw, _, _ = rel_rot.yaw_pitch_roll
        yaw = yaw  # in radians
        return x_pos, y_pos, yaw, vel, acc


