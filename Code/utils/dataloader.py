
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# nuscenes libraries
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

'''
    LOADER CLASS
    
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

    def check_consistency(self, data):
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


class NuscenesLoader(Loader):

    def __init__(self, DATAROOT='/data/sets/nuscenes', version='v1.0-mini', data_name='mini_train', verbose=True):

        # specify attributes needed in load_data() implementation
        self.version = version
        self.data_name = data_name
        self.verbose = verbose
        super(NuscenesLoader, self).__init__(DATAROOT)

    def setVerbose(self, verbose: bool):
        self.verbose = verbose

    def __get_attributes(self, instance_token:str, sample_token:str, helper:PredictHelper):
        try:
            values = helper.get_sample_annotation(instance_token, sample_token)
            abs_pos = values['translation']
            rotation = values['rotation']
            speed = helper.get_velocity_for_agent(instance_token, sample_token)
            accel = helper.get_acceleration_for_agent(instance_token, sample_token)
            heading_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            return [abs_pos[0], abs_pos[1]], rotation, speed, accel, heading_rate

        except KeyError:
            return None

    def __get_rel_pos(self, instance_token: str, nuscenes, head_sample, helper: PredictHelper):
        if head_sample['next'] != '':
            next_sample = nuscenes.get('sample', head_sample['next'])

            try:
                first_pos = helper.get_past_for_agent(instance_token, next_sample['token'], 0.5, True)
                print(first_pos)
            except KeyError:
                first_pos = np.array([])

            try:
                future_pos = helper.get_future_for_agent(instance_token, head_sample['token'], 20, True)
                print(future_pos)
            except KeyError:
                future_pos = np.array([])

            return np.append(first_pos, future_pos, axis=0)

        return []

    def load_data(self) -> dict:

        nuscenes = NuScenes(self.version, dataroot=self.DATAROOT)
        helper = PredictHelper(nuscenes)
        mini_train = get_prediction_challenge_split(self.data_name, dataroot=self.DATAROOT)

        # split instance_token from sample_token
        inst_sampl_tokens = [token.split("_") for token in mini_train]

        # dictionary of agents
        agents = {}

        # traverse all instances and samples
        for tokens in inst_sampl_tokens:
            instance_token = tokens[0]
            sample_token = tokens[1]

            # verify if agent exists
            if agents.get(instance_token) is None:
                if self.verbose:
                    print('new agent: ', instance_token, " sample: ", sample_token)

                # get sample
                sample = nuscenes.get('sample', sample_token)

                # agent does not exist, create new agent
                agents[instance_token] = {'neighbors': [],
                                          'abs_pos': [],
                                          'rel_pos': [],
                                          'rotation': [],
                                          'speed': [],
                                          'accel': [],
                                          'heading_rate': [],
                                          'context': [],
                                          'map': None}

                # get head_sample
                head_sample_token = nuscenes.get('scene', sample['scene_token'])['first_sample_token']
                head_sample = nuscenes.get('sample', head_sample_token)

                # traverse forward samples from head_sample
                tmp_sample = head_sample

                rel_pos = self.__get_rel_pos(instance_token, nuscenes, head_sample, helper)
                agents[instance_token]['rel_pos'] = rel_pos

                while tmp_sample is not None:
                    tmp_sample_token = tmp_sample['token']

                    # get abs_pos, rotation, speed, accel, heading_rate
                    attributes = self.__get_attributes(instance_token, tmp_sample_token, helper)

                    if attributes is not None:
                        # set attributes of agent
                        agents[instance_token]['context'].append(tmp_sample_token)
                        agents[instance_token]['abs_pos'].append(attributes[0])
                        agents[instance_token]['rotation'].append(attributes[1])
                        agents[instance_token]['speed'].append(attributes[2])
                        agents[instance_token]['accel'].append(attributes[3])
                        agents[instance_token]['heading_rate'].append(attributes[4])

                    # move to next sample if possible
                    try:
                        tmp_sample = nuscenes.get('sample', tmp_sample['next'])

                    except KeyError:
                        tmp_sample = None

        return agents

    def check_consistency(self, data):
        return True

    def get_custom_data_as_tensor(self, data: dict) -> tuple:
        return None


nuscenes_loader = NuscenesLoader(verbose=True)
agents = nuscenes_loader.load_data()

#trajectories = nuscenes_loader.get_trajectories_as_tensor()
#trajectories.shape


keys = [k for k in nuscenes_loader.dataset]
counter = 0
for i in range(len(keys)):
    size_abs = len(nuscenes_loader.dataset[keys[i]]['abs_pos'])
    size_rel = len(nuscenes_loader.dataset[keys[i]]['rel_pos'])
    if size_abs == size_rel:
        counter += 1
    else:
        print(size_abs, " vs ", size_rel)

print(counter)



# ------------------------------------------------------- PRUEBAS -------------------------------------------------------


# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/data/sets/nuscenes'
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)


mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
print(mini_train[:5])

inst_sampl_tokens = np.array([cad.split('_') for cad in mini_train])

# get tokens
instance_token = inst_sampl_tokens[:, 0]
sample_token = inst_sampl_tokens[:, 1]


prev_change_token = ''

for i in range(742):
    if prev_change_token != instance_token[i]:
        print(i, ": ", instance_token[i], " ", sample_token[i])
        sample = nuscenes.get('sample', sample_token[i])
        prev_sample_token = sample['prev']
        prev_sample = nuscenes.get('sample', prev_sample_token)
        annot = helper.get_sample_annotation(instance_token[i], prev_sample_token)
        prev_change_token = instance_token[i]



for i in range(len(instance_token)):
    print(helper.get_future_for_agent(instance_token[i], sample_token[i], 1, True))

len(set(instance_token))

# helper to query data
helper = PredictHelper(nuscenes)

# verify categories for instances to be tracked
for token in instance_token:
    category_token  = nuscenes.get('instance', token)['category_token']
    category = nuscenes.get('category', category_token)
    print(category['name'])


sample = helper.get_annotations_for_sample(sample_token[0])

head_sample_token = nuscenes.get('scene', sample['scene_token'])['first_sample_token']

sample = nuscenes.get('sample', sample_token[0])
prev_sample = sample['prev']

prev_sample

future_xy_local = helper.get_future_for_agent(instance_token[0], head_sample_token, seconds=1, in_agent_frame=False)
past_xy_local = helper.get_past_for_agent(instance_token[0], head_sample_token, seconds=10, in_agent_frame=False, just_xy=False)
annot = helper.get_sample_annotation(instance_token[0], nuscenes.get('sample', head_sample_token)['next'])


instance_token[0]
sample_token[0]
head_sample_token
annot = helper.get_sample_annotation('bc38961ca0ac4b14ab90e547ba79fbb6', '4711bcd34644420da8bc77163431888e')

print(nuscenes.get('sample', 'a34fabc7aa674713b71f98ec541eb2d4'))

velo = helper.get_acceleration_for_agent(instance_token[0], sample_token[1])
velo

len(future_xy_local)
len(past_xy_local)

future_xy_local
past_xy_local
annot

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)


names = [category['name'] for category in categories]
print(names)


print(len(nuscenes.sample))

def buildTestDataSet():
    #dumpFile = open("../datasets/nuscenes/mundo/mun_pos.csv", "w")

    nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
    scenes = nusc.scene

    frame_dict = {}
    counter = 1

    # traverse scenes
    for scene in scenes:
        instance_dict = {}
        instance_counter = 1
         # get first_sample
        sample_token = scene['first_sample_token']


        # traverse all samples chained to first_sample of the scene

        while sample_token != "":
            sample = nusc.get('sample', sample_token)

            annotation_tokens = sample['anns']

            for token in annotation_tokens:
                annotation_data = nusc.get('sample_annotation', token)
                instance_token = annotation_data['instance_token']

                instance = nusc.get('instance', instance_token)

                category_token = instance['category_token']
                category = nusc.get('category', category_token)
                category_name = category['name']

                if str.split(category_name, '.')[0] != "human":
                    break

                print(category_name)

                pos = annotation_data['translation']
                x_pos = pos[0]
                y_pos = pos[1]

                if frame_dict.get(sample_token) == None:
                    frame_dict[sample_token] = counter
                    counter = counter + 1

                if instance_dict.get(instance_token)==None:
                    instance_dict[instance_token] = instance_counter
                    instance_counter += 1

                if instance_dict.get(instance_token) == None:
                    print("WARN")

                line = str(frame_dict[sample_token]) + "," + str(instance_dict[instance_token]) + "," + str(x_pos) + "," + str(y_pos) + "\n"
                print(line)
                #dumpFile.write(line)

            sample_token = sample['next']

    #dumpFile.close()
    print(counter)
    return 0

buildTestDataSet()



import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

nusc_map = NuScenesMap(dataroot='/data/sets/nuscenes', map_name='singapore-onenorth')


fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)



sample_traffic_light_record = nusc_map.traffic_light[0]
sample_traffic_light_record

obj =  nuscenes.get('map', '00590fed-3542-4c20-9927-f822134be5fc')





dic = {"juan":{"pos":[1, 2, 3], "age": 15},
       "andrea":{"pos":[4, 5, 6], "age":17 },
       "luis":{"pos":[7, 8, 9], "age":20},
       "pedro":{"pos":[7, 8, 9]}}

print(dic)

print([element.get('age', None) for element in dic.values()])

