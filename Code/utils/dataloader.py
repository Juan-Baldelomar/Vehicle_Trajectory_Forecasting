
from Dataset import Dataset

# utilities
import numpy as np
import pickle


# -------------------------------------------------- READ/WRITE PKL FILES ------------------------------------------------------------

# store processed data in pkl files
def save_pkl_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        print("data stored succesfully to: ", filename)


# read processed data in pkl files
def load_pkl_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


# -------------------------------------------------------------- BASE CLASS ------------------------------------------------------

class Loader:
    """
    Loader CLASS

    * Loader Parent Class: This class should be implemented by any specific loader of the desired dataset.
    * self.dataset should be a dictionary with at least the following attributes:
        {
           'agents': {
                        <agent_key>: {
                            abs_pos: [[x_1, y_1], ... , [x_n, y_n]],
                            ego_pos: [[x_1, y_1], ... , [x_n, y_n]],
                        }
                     }
        }

      abs_pos contains the trajectory of an agent encoded as list of positions in the world coordinate system
      ego_pos contains the trajectory of the ego vehicle encoded in the same way abs_pos

    * self.dataset should be obtained when the implementarion of load_data() is called
    """

    def __init__(self, DATAROOT, verbose):
        self.DATAROOT = DATAROOT
        self.origin_offset = 0
        #self.dataset: dict = {'agents': {}, 'ego_vehicles': {}}
        self.dataset = Dataset()
        self.verbose = verbose
        self.maps = None

    def load_data(self):
        """ method to load data and should be called in the constructor"""
        raise NotImplementedError

    # store processed data in pkl files
    def save_pickle_data(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.dataset, file, pickle.HIGHEST_PROTOCOL)
            print("data stored succesfully to: ", filename)

    # read processed data in pkl files
    def load_pickle_data(self, filename):
        try:
            file = open(filename, 'rb')
            self.dataset = pickle.load(file)
            file.close()
            print('[MSG] pickle data read succesfuly from: ', filename)
            return True

        except FileNotFoundError:
            print('[WARN] file does not exist to read pickle data: ', filename)
            return False

    def get_trajectories_indexes(self, size, skip=0, mode='overlap', overlap_points=0) -> np.array:
        """
        function to get the list of pair of indexes that indicate the start and end of a trajectory. This is done because if you
        have trajectories that are much bigger than te size parameter, you may be interested in getting as many sub-trajectories
        as you can from that big trajectory.

        :param size indicates the number of points in the trajectory
        :param skip indicates the number from which the first pair of indexes of each trajectory starts. This is because some datasets might
               not have data defined in the firts observations (as nuscenes does not have speed defined in the 1st position or
               acceleration is defined until the 3rd position so it could be usefull to skip the firts 2 observations)

        :param mode indicates how to treat trajectories that are bigger than the minimum size. The following values are supported

            - 'single': truncates the trajectory to the point in the <size> position, ie. agent_trajectory[0 : size]

            - 'overlap': builds as many trajectories as it can using the overlap_points as the number of points that are common
                         between two consecutive trajectories. For example if you have a minimum size = 20, overlap_points = 10
                         and the trajectory has a lenght of 30 points, it builds two trajectories of the form [0:20], [10:30]

        :param overlap_points indicates the number of points that two consecutive trajectories can have
        :return list with tuple of indexes <(start, end)> indicating the start and end for each sub-trajectory for each agent.

        """
        for key, agent in self.dataset.agents.items():
            agent_datasize = len(agent.timesteps)

            if agent_datasize - skip >= size:
                if mode == 'overlap':
                    start, end = skip, size
                    while end <= agent_datasize:
                        agent.indexes.append((start, end))
                        start = end - overlap_points
                        end = start + size
                else:
                    agent.indexes.append((skip, size))

            elif self.verbose:
                print('Agent {} does not have enough points in the trajectory'.format(key))

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

        for (key, agent) in self.dataset['agents'].items():
            if len(agent['index'] > 0):
                agent_keys.append(key)

        return agent_keys
