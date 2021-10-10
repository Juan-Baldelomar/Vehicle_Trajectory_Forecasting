
# utilities
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle


# nuscenes libraries
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.eval.common.utils import quaternion_yaw

# map expansion libraries
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

# Tensorflow
import tensorflow as tf


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
                                rotation: [[q11, q12, q13, q14], ... , [qn1, qn2, qn3, qn4]],
                                ego_rotation: [[q11, q12, q13, q14], ... , [qn1, qn2, qn3, qn4]],
                            }
                         }
            }

          abs_pos contains the trajectory of an agent encoded as list of positions in the world coordinate system
          ego_pos contains the trajectory of the ego vehicle encoded in the same way abs_pos
          rotation contains  a list of the rotations encoded as a quaternion
          ego_rotation contains a list of the rotations of the ego_vehicle encoded as quaternion

        * self.dataset should be obtained when the implementarion of load_data() is called
    """

    def __init__(self, DATAROOT):
        self.DATAROOT = DATAROOT
        self.origin_offset = 0
        self.dataset: dict = {'agents': {}}

    def load_data(self):
        """ method to load data and should be called in the constructor"""
        raise NotImplementedError

    def check_consistency(self):
        """check that the data was load propperly and it fulfills the desired characteristics"""
        raise NotImplementedError

    def get_custom_data_as_tensor(self):
        """ Load custom data as tensor. This functions deals with the specific details of each dataset to get the data
            in the desired format as a tensor. It is supposed to deal with specific context attributes of each dataset
        """
        raise NotImplementedError

    # store processed data in pkl files
    def save_pickle_data(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self.dataset, file, pickle.HIGHEST_PROTOCOL)
        file.close()
        print("data stored succesfully to: ", filename)

    # read processed data in pkl files
    def load_pickle_data(self, filename):
        try:
            file = open(filename, 'rb')
            self.dataset = pickle.load(file)
            file.close()
            return True

        except FileNotFoundError:
            print('[WARN] file does not exist to read pickle data: ', filename)
            return False

    def get_trajectories_indexes(self, size=20, mode='overlap', overlap_points=0) -> np.array:
        """
                function to get the list of indexes that indicate the start and end of a trajectory. This is done because if you
                have trajectories that are much bigger than te size parameter, you may be interested in getting as many sub-trajectories
                as you can from that big trajectory.

                :param size indicates the number of points in the trajectory
                :param mode indicates how to treat trajectories that are bigger than the minimum size. The following values are supported

                    - 'single': truncates the trajectory to the point in the <size> position, ie. agent_trajectory[0 : size]

                    - 'overlap': builds as many trajectories as it can using the overlap_points as the number of points that are common
                                 between two consecutive trajectories. For example if you have a minimum size = 20, overlap_points = 10
                                 and the trajectory has a lenght of 30 points, it builds two trajectories of the form [0:20], [10:30]

                :param overlap_points indicates the number of points that two consecutive trajectories can have
                :return list with tuple of indexes <(start, end)> indicating the start and end for each sub-trajectory for each agent.

        """

        for agent in self.dataset['agents'].values():
            agent_datasize = len(agent['abs_pos'])
            agent['index'] = []

            if agent_datasize >= size:
                if mode == 'overlap':
                    start, end = 0, size
                    while end < agent_datasize:
                        agent['index'].append((start, end))
                        start = end - overlap_points
                        end = start + size
                else:
                    agent['index'].append((0, size))

    def get_trajectories_data(self, size=20, mode='overlap', overlap_points=0, offset=10) -> np.array:
        f"""
                get absolute positons, rotations, relative positions and relative rotations to a given point in the ego_positions 
                and a given rotation of the  trajectories 
                
                :param size indicates the minimum size of points in the trajectory (see get_trajectories_indexes doc)
                :param mode indicates how to treat trajectories that are bigger than the minimum size (see get_trajectories_indexes doc)
                :param overlap_points indicates the number of points that two consecutive sub-trajectories can have
                :param offset indicates which point of the ego positions will be taken as the origin
                
                :return dictionary containing the desired values as numpy arrays
        """
        self.origin_offset = offset
        self.get_trajectories_indexes(size, mode, overlap_points)

        key_agents = self.dataset['agents'].items()
        trajectories = {'id': [], 'abs_pos': [], 'rel_pos': [], 'rotation': [], 'rel_rotation': []}

        for (key, agent) in key_agents:
            indexes = agent['index']
            for index in indexes:
                start, end = index

                # get data to process
                abs_pos = agent['abs_pos'][start: end]
                rotation = agent['rotation'][start: end]
                ego_pos = agent['ego_pos'][start: end]
                ego_rotation = agent['ego_rotation'][start: end]

                # get relative positions
                origin = ego_pos[offset]
                rel_pos = abs_pos - origin

                # get relative rotations
                ego_rotation_axis_quaternion = Quaternion(ego_rotation[offset])
                eraq_inv = ego_rotation_axis_quaternion.inverse
                rel_rotation = [(eraq_inv * Quaternion(rot)).elements for rot in rotation]

                # append values
                trajectories['id'].append(key)
                trajectories['abs_pos'].append(abs_pos)
                trajectories['rotation'].append(rotation)
                trajectories['rel_pos'].append(rel_pos)
                trajectories['rel_rotation'].append(rel_rotation)

        return trajectories


class NuscenesLoader(Loader):
    """
        NuscenesLoader CLASS

        * This class is the specific implementation for the loader of the nuscenes dataset
        * self.dataset is a dictionary with the following attributes:
            {
               agents:{
                    <agent_key>:{
                                    abs_pos: [],                # list of coordinates in the world frame (2d)
                                    ego_pos: [],                # list of coordinates in the vehicle with the camera frame (2d)
                                    rotation: [],               # list of orientation of the annotation box parametrized as a quaternion
                                    ego_rotation: [],           # list of orientation of the annotation box parametrized as a quaternion
                                    speed: [],                  # list of angular speed (scalar, defined from the second annotation of the instance
                                                                  not the first one)
                                    accel: [],                  # list of acceleration (scalar, defined from the third annotation of the instance
                                    heading_rate: [],
                                    context: [],                # list of context_ids, those ids point to anothe dictionary with relevan information
                                                                  of the context
                                }
                    }
               context:{
                    <context_key> (same as sample token) :
                                {
                                    neighbors: []               # list of instance neighbors that are vehicles in the sample
                                    humans: []                  # list of pedestrians in the sample
                                    objects: []                 # list of objects in the sample
                                    map: str                    # map id of the sample
                                }
               }
            }

        * self.dataset dictionary should be filled with the implementarion of load_data()

        * IMPORTANT:
            1. a SCENE is made by several moments in time or time steps. This time steps are called SAMPLES.
            2. It is relevant to know that the information of an agent (INSTANCE table) in time is stored in the SAMPLE_ANNOTATION table.
               In a database context this would be the 'MASTER' table. An agent(instance table) has several recordings in time (sample table)
               and a sample has several agents in it, so a master table is needed, therefore the sample_annotation table fullfills this.
    """

    def __init__(self, DATAROOT='/data/sets/nuscenes', pickle=True, pickle_filename='/data/sets/nuscenes/pickle/nuscenes_data.pkl',
                 version='v1.0-mini', data_name='mini_train', verbose=True, rel_offset=10):

        # parent constructor
        super(NuscenesLoader, self).__init__(DATAROOT)

        # specify nuscenes attributes
        self.version: str = version
        self.data_name: str = data_name
        self.nuscenes = NuScenes(version, dataroot=DATAROOT)
        self.helper = PredictHelper(self.nuscenes)
        self.verbose: bool = verbose
        self.rel_offset = rel_offset

        # nuscenes map expansion attributes
        # map names = 'singapore-onenorth', 'singepore-hollandvillage', 'singapore-queenstown', 'boston-seaport'

        self.maps = {
            'singapore-onenorth': NuScenesMap(dataroot=DATAROOT, map_name='singapore-onenorth'),
            'singapore-hollandvillage': NuScenesMap(dataroot=DATAROOT, map_name='singapore-hollandvillage'),
            'singapore-queenstown': NuScenesMap(dataroot=DATAROOT, map_name='singapore-queenstown'),
            'boston-seaport': NuScenesMap(dataroot=DATAROOT, map_name='boston-seaport')
        }

        # flag to indicate if data can be loaded from pickle files
        pickle_ok: bool = os.path.isfile(pickle_filename)

        if pickle and pickle_ok:
            # its okay to read pickle files to load data
            self.load_pickle_data(pickle_filename)

        else:
            # load data from scratch
            self.dataset['context'] = {}
            self.dataset['agents'] = self.load_data()
            self.get_context_information()
            self.save_pickle_data(pickle_filename)

    # set verbose mode which determines if it should print the relevant information while processing the data
    def setVerbose(self, verbose: bool):
        self.verbose = verbose

    # create and insert neighbors to the context dictionary
    def insert_context_neighbor(self, instance_token: str, sample_token: str):

        context: dict = self.dataset['context']

        if context.get(sample_token) is None:

            # get the location (map name)
            scene_token = self.nuscenes.get('sample', sample_token)['scene_token']
            scene = self.nuscenes.get('scene', scene_token)
            location = self.nuscenes.get('log', scene['log_token'])['location']

            # build entry in the dictionary
            context[sample_token] = {
                                        'neighbors':        [],
                                        'humans':           [],
                                        'objects':          [],
                                        'map':              location
                                    }

        context[sample_token]['neighbors'].append(instance_token)

    # get attributes of and agent in a specific moment in time (sample)
    def __get_agent_attributes(self, sample_annotation) -> tuple:

        # get tokens
        sample_token = sample_annotation['sample_token']
        instance_token = sample_annotation['instance_token']

        # get sample
        sample = self.nuscenes.get('sample', sample_token)

        # get attributes
        abs_pos = sample_annotation['translation']
        rotation = sample_annotation['rotation']
        speed = self.helper.get_velocity_for_agent(instance_token, sample_token)
        accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        heading_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        # get sample_data_token of a sensor to read EGO_POSE
        sample_data_token = sample['data']['RADAR_FRONT']

        # get ego_pose
        ego_pose_token = self.nuscenes.get('sample_data', sample_data_token)['ego_pose_token']
        ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)
        ego_pose_xy = ego_pose['translation'][:2]
        ego_rotation = ego_pose['rotation']

        return [abs_pos[0], abs_pos[1]], rotation, speed, accel, heading_rate, ego_pose_xy, ego_rotation

    # <NOT USED> get all the relative positions of an agent in the scene (relative to the vehicle that has the camera recording)
    def __get_rel_pos(self, instance_token: str, head_sample: dict):

        # to get the first relative position you need to move to the next sample to see the past (this is due to how the data is stored)
        next_sample = self.nuscenes.get('sample', head_sample['next'])

        # see the past to get the first relative position
        first_pos = self.helper.get_past_for_agent(instance_token, next_sample['token'], 0.5, True)

        # see the future to get all the other relative positions
        future_pos = self.helper.get_future_for_agent(instance_token, head_sample['token'], 20, True)
        return np.append(first_pos, future_pos, axis=0)

    def load_data(self) -> dict:
        # list of the form <instance_token>_<sample_token>
        mini_train: list = get_prediction_challenge_split(self.data_name, dataroot=self.DATAROOT)

        # split instance_token from sample_token
        inst_sampl_tokens = np.array([token.split("_") for token in mini_train])
        instance_tokens = set(inst_sampl_tokens[:, 0])

        # dictionary of agents
        agents = {}

        # traverse all instances and samples
        for instance_token in instance_tokens:
            instance = self.nuscenes.get('instance', instance_token)

            # verify if agent exists
            if agents.get(instance_token) is None:
                if self.verbose:
                    print('new agent: ', instance_token)

                # agent does not exist, create new agent
                agents[instance_token] = {'abs_pos':        [],                # list of coordinates in the world frame (2d) (sequence)
                                          'ego_pos':        [],                # position of the ego vehicle, the one with the cameras
                                          'rotation':       [],                # list of rotation parametrized as a quaternion
                                          'ego_rotation':   [],                # list of rotations of ego vehicle
                                          'speed':          [],                # list of scalar
                                          'accel':          [],                # list scalar
                                          'heading_rate':   [],                # list scalar
                                          'context':        [],                # list of ids
                                          }

                agent = agents[instance_token]

                # get head_annotation
                first_annotation_token: str = instance['first_annotation_token']
                first_annotation: dict = self.nuscenes.get('sample_annotation', first_annotation_token)

                # tmp annotation to traverse them
                tmp_annotation = first_annotation

                # GET and SET all the relative positions (relative to the camera in the vehicle)
                #rel_pos: numpy.array = self.__get_rel_pos(instance_token, self.nuscenes.get('sample', first_annotation['sample_token']))
                #agent['rel_pos'] = rel_pos

                # traverse forward sample_annotations from first_annotation
                while tmp_annotation is not None:
                    sample_token = tmp_annotation['sample_token']

                    # insert neighbors to corresponding context dictionary
                    self.insert_context_neighbor(instance_token, sample_token)

                    # GET abs_pos: [], rotation: [], speed: float , accel: float, heading_rate: float, ego_pose: np.array([x, y]),
                    # ego_rotation : np.array([z1, z2, z3, z4])
                    attributes: tuple = self.__get_agent_attributes(tmp_annotation)

                    # set attributes of agent
                    agent['context'].append(sample_token)
                    agent['abs_pos'].append(attributes[0])
                    agent['rotation'].append(attributes[1])
                    agent['speed'].append(attributes[2])
                    agent['accel'].append(attributes[3])
                    agent['heading_rate'].append(attributes[4])
                    agent['ego_pos'].append(attributes[5])
                    agent['ego_rotation'].append(attributes[6])

                    # move to next sample_annotation if possible
                    try:
                        tmp_annotation = self.nuscenes.get('sample_annotation', tmp_annotation['next'])

                    except KeyError:
                        tmp_annotation = None

                # close while

                # transform to np arrays
                agent['abs_pos'] = np.array(agent['abs_pos'])
                agent['ego_pos'] = np.array(agent['ego_pos'])
                agent['rotation'] = np.array(agent['rotation'])
                agent['ego_rotation'] = np.array(agent['ego_rotation'])

            # close IF agent.get(instance_token) Block

        # close For instance_token
        return agents

    def get_context_information(self):
        """
            function to get the context information. We ought to rember that context in this dataset is obtained from the
            SAMPLE_ANNOTATION table. By the moment we obtain the pedestrians and obstacles in a record of sample_annotation
            (a record of an agent in time) and we store the SAMPLE_ANNOTATION TOKEN so we can then obtain any information that
            is relevant to the model, like its position, its attributes, etc, by quering the database of this record.

            :return: None
        """
        dataset_context = self.dataset['context']
        keys = dataset_context.keys()
        for key in keys:
            sample = self.nuscenes.get('sample', key)
            context: dict = dataset_context[key]
            sample_annotation_tokens = sample['anns']

            # get context information
            for ann_token in sample_annotation_tokens:
                sample_annotation = self.nuscenes.get('sample_annotation', ann_token)

                # get the instance category
                categories = sample_annotation['category_name'].split('.')

                # add the token to the corresponding list if necessary
                if categories[0] == 'human':
                    context['humans'].append(sample_annotation['token'])
                elif categories[0] == ('movable_object' or 'static_object'):
                    context['objects'].append(sample_annotation['token'])

    def check_consistency(self):
        """
        function to check if the data obtained is consistent. In the nuscenes dataset, values in the agent dictionary
        should have the same amount of observations.

        :return: bool
        """
        # set flag by default to true
        flag = True

        keys = self.dataset['agents'].keys()
        # traverse agents
        for key in keys:
            agent = self.dataset['agents'][key]
            size_abs = len(agent['abs_pos'])
            size_rel = len(agent['ego_pos'])
            size_speed = len(agent['speed'])
            size_accel = len(agent['accel'])
            size_context = len(agent['context'])
            size_rotation = len(agent['rotation'])
            size_ego_rotation = len(agent['ego_rotation'])
            size_heading_rate = len(agent['heading_rate'])

            #  verify all the fields of the agent have the same size (same number of data for each moment in time)
            if not (size_abs == size_rel == size_speed == size_accel == size_context == size_rotation
                    == size_ego_rotation == size_heading_rate):
                flag = False
                print('[WARN]: no consistency with obs ->', key)

        return flag

    def plotMasks(self, agent: dict, height=200, width=200):
        """
        exploratory function to plot the bitmaps of an agent's positions
        :param agent: agent's data dictionary
        :param height: height of the bitmap
        :param width:  width of the bitmap
        :return: None
        """

        # get map
        context_id = agent['context'][0]
        map_name = self.dataset['context'][context_id]['map']
        map = self.maps[map_name]

        # traverse agent positions
        for pos in agent['abs_pos']:
            x, y = pos[0], pos[1]
            patch_box = (x, y, height, width)
            patch_angle = 0  # Default orientation where North is up
            layer_names = ['drivable_area', 'walkway']
            canvas_size = (1000, 1000)

            figsize = (12, 4)
            fig, ax = map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=figsize, n_row=1)
            fig.show()

    def getMasks(self, agent, height=200, width=200):
        """
         function to get the bitmaps of an agent's positions
        :param agent: agent's data dictionary
        :param height: height of the bitmap
        :param width:  width of the bitmap
        :return: list of bitmaps (each mask contains 2 bitmaps)
        """

        # get map
        context_id = agent['context'][0]
        map_name = self.dataset['context'][context_id]['map']
        map = self.maps[map_name]

        masks_list = []

        # traverse agents positions
        for pos in agent['abs_pos']:
            x, y = pos[0], pos[1]
            patch_box = (x, y, height, width)
            patch_angle = 0  # Default orientation where North is up
            layer_names = ['drivable_area', 'walkway']
            canvas_size = (1000, 1000)
            map_mask = map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)
            masks_list.append(map_mask)

        return masks_list

    def get_custom_data_as_tensor(self, data: dict) -> tuple:
        return None


# -------------------------------------------------------------------- TESTING -------------------------------------------------------------------- #


dataroot_base = '/data/sets/nuscenes'
dataroot_train = '/media/juan/Elements'

# dataset attributes
#dataroot = dataroot_train + dataroot_base
dataroot = dataroot_base

#version = 'v1.0-trainval'
version = 'v1.0-mini'

#data_name = 'train'
data_name = 'mini_train'

nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name)
print(nuscenes_loader.check_consistency())

trajectories_dataset = nuscenes_loader.get_trajectories_data(mode='overlap')


# get tensorflow dataset
pos_dataset = tf.data.Dataset.from_tensor_slices(trajectories_dataset['abs_pos'])
rot_dataset = tf.data.Dataset.from_tensor_slices(trajectories_dataset['rotation'])

full_dataset = tf.data.Dataset.zip((pos_dataset, rot_dataset))

pos_element = pos_dataset.take(1)
rot_element = rot_dataset.take(1)
full_element = full_dataset.take(1)

for i, x in enumerate(pos_element):
    print(i, " ", x)

for i, x in enumerate(rot_element):
    print(i, " ", x)

for i, (x, y) in enumerate(full_element):
    print(tf.concat((x, y), axis=1))


agents_list = list(nuscenes_loader.dataset['agents'].values())


save_pkl_data(trajectories_dataset, '/data/traj_data.pkl')
data = load_pkl_data('/data/traj_data.pkl')
# -------------------------------------------------------------------- MAPA --------------------------------------------------------------------


# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = dataroot

# nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
# helper = PredictHelper(nuscenes)
# mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
# toks = [x.split('_') for x in mini_train]

#nusc_map = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
#fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=2)
#fig.show()


#nuscenes_loader.plotMasks(agents_list[0])

# -------------------------------------------------------------------- exploratory --------------------------------------------------------------------


# j = 10
# agent_id = trajectories_dataset['id'][j]
# context_id = nuscenes_loader.dataset['agents'][agent_id]['context'][0]
# map_name = nuscenes_loader.dataset['context'][context_id]['map']
#
# ego_pos = nuscenes_loader.dataset['agents'][agent_id]['ego_pos'][10]
# ego_rot = nuscenes_loader.dataset['agents'][agent_id]['ego_rotation'][10]
# rel_rot = trajectories_dataset['rel_rotation'][j][10]
# rotation = trajectories_dataset['rotation'][j][10]
#
# print('ego: ', ego_rot)
# print('rot: ', rotation)
# print('rel_rot: ', rel_rot)
#
# nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
# #fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=2)
#
# x, y = ego_pos[0], ego_pos[1]
# patch_box = (x, y, 200, 200)
#
# patch_angle = []
# patch_angle.append(0)
# patch_angle.append(quaternion_yaw(Quaternion(ego_rot)))
# patch_angle.append(quaternion_yaw(Quaternion(list(rotation))))
# patch_angle.append(quaternion_yaw(Quaternion(list(rel_rot))))
#
#
# layer_names = ['drivable_area', 'walkway']
# canvas_size = (1000, 1000)
#
# figsize = (12, 4)
# for i in range(4):
#     fig, ax = nusc_map.render_map_mask(patch_box, patch_angle[i], layer_names, canvas_size, figsize=figsize, n_row=1)
#     fig.show()
#
#
# q1 = Quaternion([0.475, 0, 0, 0.88])
# q2 = Quaternion([0.711, 0.0, 0.0, 0.703])
# q3 = q1.inverse * q2
#
# q4 = q3.inverse * q2
#
# q6 = q3.inverse * q1
# q5 = Quaternion([ 0.956, 0, 0, -0.292])
#
# q2.angle
