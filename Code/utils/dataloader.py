
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# nuscenes libraries
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

# map expansion libraries
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

'''
    Loader CLASS
    
    * Loader Parent Class: This class should be implemented by any specific loader of the desired dataset.
    * self.dataset should be a dictionary with at least the following attributes:
        { 
            <agent_key> : { 
                abs_pos: [[x_1, y_1], ... , [x_n, y_n]],
            } 
        }
    
      abs_pos contains the trajectory encoded as list of positions in the world coordinate system
    
    * self.dataset is obtained when the implementarion of load_data() is called
    * super().__init__ should be called from the child constructor when all the self.<attributes> needed in the implementation
      of load_data() are already specified. 
'''


class Loader:

    def __init__(self, DATAROOT):
        self.DATAROOT = DATAROOT
        self.dataset: dict = self.load_data()           # calls for specific implemented class load_data function

    # private method, should be called in the constructor
    def load_data(self):
        raise NotImplementedError

    def check_consistency(self):
        raise NotImplementedError

    def get_custom_data_as_tensor(self):
        raise NotImplementedError

    '''
        function to get the trajectories from all the agents in the dataset as a tensor
        @:param data contains the data as a dictionary where the trajectories are loaded
        @:param size indicates the minimum size of points in the trajectory
        
        * To be able to use the data as a tensor all trajectories must keep the same size,
          so trajectories that are longer in points will be truncated from position 0 to positon <size>
    '''

    def get_trajectories_as_tensor(self, size=20) -> np.array:

        trajectories = [agent['abs_pos'] for agent in self.dataset.values()]
        filtered_trajectories = []
        for agent_traj in trajectories:

            if len(agent_traj) >= size:
                filtered_trajectories.append(agent_traj[:size])

        return np.array(filtered_trajectories)


'''
    NuscenesLoader CLASS

    * This class is the specific implementation for the loader of the nuscenes dataset
    * self.dataset is a dictionary with the following attributes:
        { 
            <agent_key>:    {
                                neighbors: [],              # list of neighbors of the agent
                                abs_pos: [],                # list of coordinates in the world frame (2d)
                                rel_pos: [],                # list of coordinates in the vehicle with the camera frame (2d)
                                rotation: [],               # list of orientation of the annotation box parametrized as a quaternion
                                speed: [],                  # list of angular speed (scalar, defined from the second annotation of the instance
                                                              not the first one)
                                accel: [],                  # list of acceleration (scalar, defined from the third annotation of the instance
                                heading_rate: [],
                                context: [],                # list of context_ids, those ids point to anothe dictionary with relevan information
                                                              of the context
                                map: None
                            }
        }

    * self.dataset is obtained when the implementarion of load_data() is called
'''


class NuscenesLoader(Loader):

    def __init__(self, DATAROOT='/data/sets/nuscenes', version='v1.0-mini', data_name='mini_train', verbose=True):

        # specify attributes needed in load_data() implementation
        self.version = version
        self.data_name = data_name
        self.verbose = verbose

        # note that the parent constructor is called after all the attributes needed in the load_data() function are already set.
        super(NuscenesLoader, self).__init__(DATAROOT)

    def setVerbose(self, verbose: bool):
        self.verbose = verbose

    def __get_attributes(self, sample_annotation, helper: PredictHelper):
        sample_token = sample_annotation['sample_token']
        instance_token = sample_annotation['instance_token']

        abs_pos = sample_annotation['translation']
        rotation = sample_annotation['rotation']
        speed = helper.get_velocity_for_agent(instance_token, sample_token)
        accel = helper.get_acceleration_for_agent(instance_token, sample_token)
        heading_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        return [abs_pos[0], abs_pos[1]], rotation, speed, accel, heading_rate

    def __get_rel_pos(self, instance_token: str, nuscenes, head_sample, helper: PredictHelper):
        next_sample = nuscenes.get('sample', head_sample['next'])

        first_pos = helper.get_past_for_agent(instance_token, next_sample['token'], 0.5, True)
        future_pos = helper.get_future_for_agent(instance_token, head_sample['token'], 20, True)
        return np.append(first_pos, future_pos, axis=0)

    def load_data(self) -> dict:
        nuscenes = NuScenes(self.version, dataroot=self.DATAROOT)
        helper = PredictHelper(nuscenes)
        mini_train = get_prediction_challenge_split(self.data_name, dataroot=self.DATAROOT)

        # split instance_token from sample_token
        inst_sampl_tokens = np.array([token.split("_") for token in mini_train])
        instance_tokens = set(inst_sampl_tokens[:, 0])

        # dictionary of agents
        agents = {}

        # traverse all instances and samples
        for instance_token in instance_tokens:
            instance = nuscenes.get('instance', instance_token)

            # verify if agent exists
            if agents.get(instance_token) is None:
                if self.verbose:
                    print('new agent: ', instance_token)

                # agent does not exist, create new agent
                agents[instance_token] = {'neighbors': [],
                                          'abs_pos': [],                # coordinates in the world frame (2d)
                                          'rel_pos': [],                # coordinates in the agent frame (2d)
                                          'rotation': [],
                                          'speed': [],                  #
                                          'accel': [],
                                          'heading_rate': [],
                                          'context': [],
                                          'map': None
                                          }
                # get head_annotation
                first_annotation_token = instance['first_annotation_token']
                first_annotation = nuscenes.get('sample_annotation', first_annotation_token)

                # traverse forward sample_annotations from first_annotation
                tmp_annotation = first_annotation

                rel_pos = self.__get_rel_pos(instance_token, nuscenes, nuscenes.get('sample', first_annotation['sample_token']), helper)
                agents[instance_token]['rel_pos'] = rel_pos

                while tmp_annotation is not None:
                    sample_token = tmp_annotation['sample_token']

                    # get abs_pos, rotation, speed, accel, heading_rate
                    attributes = self.__get_attributes(tmp_annotation, helper)

                    # set attributes of agent
                    agents[instance_token]['context'].append(sample_token)
                    agents[instance_token]['abs_pos'].append(attributes[0])
                    agents[instance_token]['rotation'].append(attributes[1])
                    agents[instance_token]['speed'].append(attributes[2])
                    agents[instance_token]['accel'].append(attributes[3])
                    agents[instance_token]['heading_rate'].append(attributes[4])

                    # move to next sample_annotation if possible
                    try:
                        tmp_annotation = nuscenes.get('sample_annotation', tmp_annotation['next'])

                    except KeyError:
                        tmp_annotation = None

        return agents

    def check_consistency(self):
        keys = [k for k in self.dataset]
        flag = True
        for i in range(len(keys)):
            size_abs = len(self.dataset[keys[i]]['abs_pos'])
            size_rel = len(self.dataset[keys[i]]['rel_pos'])
            size_speed = len(self.dataset[keys[i]]['speed'])
            size_accel = len(self.dataset[keys[i]]['accel'])
            size_context = len(self.dataset[keys[i]]['context'])
            size_rotation = len(self.dataset[keys[i]]['rotation'])
            size_heading_rate = len(self.dataset[keys[i]]['heading_rate'])

            if size_abs + size_rel + size_speed + size_accel + size_context + size_rotation + size_heading_rate != 7 * size_abs:
                flag = False
                print('[WARN]: no consistency with obs ->', keys[i])

        return flag

    def get_custom_data_as_tensor(self, data: dict) -> tuple:
        return None


nuscenes_loader = NuscenesLoader(verbose=True)
print(nuscenes_loader.check_consistency())

trajectories = nuscenes_loader.get_trajectories_as_tensor()
trajectories.shape


# ------------------------------------------------------- PRUEBAS -------------------------------------------------------


# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/data/sets/nuscenes'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
print(mini_train[:5])

# helper to query data
helper = PredictHelper(nuscenes)

inst_sampl_tokens = np.array([cad.split('_') for cad in mini_train])



# get tokens
instance_token = inst_sampl_tokens[:, 0]
sample_token = inst_sampl_tokens[:, 1]


prev_change_token = ''
count = 0
for i in range(742):
    if prev_change_token != instance_token[i]:
        print(i, ": ", instance_token[i], " ", sample_token[i])
        sample = nuscenes.get('sample', sample_token[i])
        prev_sample_token = sample['prev']
        prev_sample = nuscenes.get('sample', prev_sample_token)
        annot = helper.get_sample_annotation(instance_token[i], prev_sample_token)
        past = helper.get_past_for_agent(instance_token[i], sample_token[i], 2, False)

        if len(past) >= 1:
            print("WARN: ",  instance_token[i], " ", sample_token[i], " past: ", past)
            count += 1
        prev_change_token = instance_token[i]

count

for i in range(652, 671):
    print(instance_token[i], " ", sample_token[i])

k = 653
sample = nuscenes.get('sample', sample_token[k])
prev_sample_token = sample['prev']
prev_sample = nuscenes.get('sample', prev_sample_token)
annot = helper.get_sample_annotation(instance_token[k], prev_sample_token)
sample = prev_sample

annot



# verify categories for instances to be tracked
for token in instance_token:
    category_token = nuscenes.get('instance', token)['category_token']
    category = nuscenes.get('category', category_token)
    print(category['name'])



nusc_map = NuScenesMap(dataroot='/data/sets/nuscenes', map_name='singapore-onenorth')
fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)

sample_traffic_light_record = nusc_map.traffic_light[0]
sample_traffic_light_record

obj =  nuscenes.get('map', '00590fed-3542-4c20-9927-f822134be5fc')
