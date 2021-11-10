

class Agent:
    def __init__(self):
        self.abs_pos = []           # list of coordinates in the world frame (2d) (sequence)
        self.ego_pos = []           # position of the ego vehicle, the one with the cameras
        self.rotation = []          # list of rotation parametrized as a quaternion
        self.ego_rotation = []      # list of rotations of ego vehicle
        self.speed = []             # list of scalar
        self.accel = []             # list scalar
        self.heading_rate = []      # list scalar
        self.context = {'next': 0}  # dictionary of ids of context-scenes.
        self.map_name = None

    def plotMasks(self, maps: dict, height=200, width=200):
        """
        exploratory function to plot the bitmaps of an agent's positions
        :param agent: agent's data dictionary
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
        :param agent: agent's data dictionary
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
