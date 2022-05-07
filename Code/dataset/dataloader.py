
from Code.dataset.DataModel import Dataset

# utilities
import numpy as np
import pickle


# -------------------------------------------------------------- BASE CLASS ------------------------------------------------------

class Loader:
    """
    Loader CLASS

    * Loader Parent Class: This class should be implemented by any specific loader of the desired dataset.
    * self.dataset should be an object Datset defined in the DataModel.py. The ideal is that it contains
            - NuscenesAgent dictionary like agents[agent_id] = NuscenesAgent() object.
            - EgoVehicles dictionary like ego_vehicles[ego_id] = NuscenesEgoVehicle() object.
            - Context dictionary like Context[context_id] = Context Object. A context is an object that holds information (Ideally)
              of the neighbors (candidates of prediction) and all the other agents (like persons, obstacles, etc...) in a Sample from the
              scene. A Sample of the scene is a timestep of the scene that contains information stored in the database.
            -NOTE: separation between NuscenesEgoVehicle and NuscenesAgent is made because some datasets like nuscenes might not annotate the same
             information for the ego vehicle and an agent.

    * self.dataset should be obtained when the implementarion of load_data() is called
    """

    def __init__(self, DATAROOT, verbose):
        self.DATAROOT = DATAROOT
        self.origin_offset = 0
        #self.dataset: dict = {'agents': {}, 'ego_vehicles': {}}
        self.dataset = Dataset()
        self.verbose = verbose
        self.maps = None

    def load_data(self, *args):
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
