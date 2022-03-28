import matplotlib.pyplot as plt

nusc_ends = {'singapore-onenorth': (1500, 2000),
             'singapore-queenstown': (3200, 3500),
             'singapore-hollandvillage': (2700, 3000),
             'boston-seaport': (3000, 2200)}


class AgentTimestep:
    def __init__(self, x, y, rot, speed, accel, heading_rate, ego_pos_x, ego_pos_y, ego_rot):
        self.x = x
        self.y = y
        self.rot = rot
        self.speed = speed
        self.accel = accel
        self.heading_rate = heading_rate
        self.ego_pos_x = ego_pos_x
        self.ego_pos_y = ego_pos_y
        self.ego_rot = ego_rot


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
        self.ego_id = ego_id
        self.indexes = []
        self.map_name = map_name
        self.ego_steps = {}             # dict of steps in time

    def add_step(self, step_id, ego_step):
        if self.ego_steps.get(step_id) is None:
            self.ego_steps[step_id] = ego_step

    def get_neighbors(self, context_pool):
        neighbors_across_time = set()
        for timestep_id, ego_step in self.ego_steps.items():
            for neighbor in context_pool[timestep_id].neighbors:
                neighbors_across_time.add(neighbor)

        return neighbors_across_time

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
        self.contexts: Context = {}
        self.ego_vehicles: EgoVehicle = {}

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
