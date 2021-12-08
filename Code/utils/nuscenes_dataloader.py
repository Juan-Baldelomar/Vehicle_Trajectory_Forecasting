
# base class
from dataloader import Loader

# tools
import os.path

# data manipulation and graphics
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent

# nuscenes libraries
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.eval.common.utils import quaternion_yaw

# map expansion libraries
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

# -------------------------------------------------------------- NUSCENES LOADER CLASS -----------------------------------------------------------


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
                                ego_rotation: [],           # list of orientation of the ego_vehicle parametrized as a quaternion
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
                                neighbors: []               # list of instance neighbors that are VEHICLES in the sample
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
        super(NuscenesLoader, self).__init__(DATAROOT, verbose)

        # specify nuscenes attributes
        self.version: str = version
        self.data_name: str = data_name
        self.nuscenes = NuScenes(version, dataroot=DATAROOT)
        self.helper = PredictHelper(self.nuscenes)
        #self.verbose: bool = verbose
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

        Agent.context_dict = self.dataset['context']

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

    def load_data(self) -> dict:
        # list of the form <instance_token>_<sample_token>
        mini_train: list = get_prediction_challenge_split(self.data_name, dataroot=self.DATAROOT)

        # split instance_token from sample_token
        inst_sampl_tokens = np.array([token.split("_") for token in mini_train])
        instance_tokens = set(inst_sampl_tokens[:, 0])

        # dictionary of agents
        agents: dict[Agent] = {}

        # traverse all instances and samples
        for instance_token in instance_tokens:
            instance = self.nuscenes.get('instance', instance_token)

            # verify if agent does not exist, if it does information was already retrieved
            if agents.get(instance_token) is None:
                if self.verbose:
                    print('new agent: ', instance_token)

                # agent does not exist, create new agent
                agents[instance_token] = Agent()
                agent: Agent = agents[instance_token]

                # get head_annotation
                first_annotation_token: str = instance['first_annotation_token']
                first_annotation: dict = self.nuscenes.get('sample_annotation', first_annotation_token)

                # tmp annotation to traverse them
                tmp_annotation = first_annotation

                # get the map_name of the agent
                scene_token = self.nuscenes.get('sample', first_annotation['sample_token'])['scene_token']
                scene = self.nuscenes.get('scene', scene_token)
                location = self.nuscenes.get('log', scene['log_token'])['location']

                agent.map_name = location

                # traverse forward sample_annotations from first_annotation
                while tmp_annotation is not None:
                    sample_token = tmp_annotation['sample_token']

                    # insert neighbors to corresponding context dictionary
                    self.insert_context_neighbor(instance_token, sample_token)

                    # get agent attributes tuple as: [0]-> abs_pos: list, [1]-> rotation: list, [2]-> speed: float ,
                    # [3]-> accel: float, [4]-> heading_rate: float, [5]-> ego_pose: list, [6]-> ego_rotation : list
                    attributes: tuple = self.__get_agent_attributes(tmp_annotation)

                    # set attributes of agent
                    agent.add_observation(sample_token, attributes[0], attributes[1], attributes[2], attributes[3],
                                          attributes[4], attributes[5], attributes[6])

                    # move to next sample_annotation if possible
                    try:
                        tmp_annotation = self.nuscenes.get('sample_annotation', tmp_annotation['next'])

                    except KeyError:
                        tmp_annotation = None

                # close while

                # transform to np arrays
                agent.abs_pos = np.array(agent.abs_pos)
                agent.ego_pos = np.array(agent.ego_pos)
                agent.rotation = np.array(agent.rotation)
                agent.ego_rotation = np.array(agent.ego_rotation)

            # close IF agent.get(instance_token) Block

        # close For instance_token
        return agents

    def get_context_information(self):
        """
        function to get the context information. We ought to rember that context in this dataset is obtained from the
        SAMPLE_ANNOTATION table. By the moment we obtain the pedestrians and obstacles information in a record of sample_annotation
        (a record of an agent in time) and we store the SAMPLE_ANNOTATION TOKEN so we can then obtain any information that
        is relevant to the model, like its position, its attributes, etc, by quering the database of this record.

        :return: None
        """

        def add_non_agent_info(collection: dict, sample_annotation):
            sample_annotation_token = sample_annotation['token']
            if collection.get(sample_annotation_token) is None:
                collection[sample_annotation_token] = {
                    'abs_pos': sample_annotation['translation'][0:2]
                }

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

    def _get_custom_trajectories_data(self, **kwargs):
        """
        get absolute positons, rotations, relative positions and relative rotations to a given point in the ego_positions
        and a given rotation of the  trajectories

        :param kwargs should contain the parameter 'offset' indicates which point of the ego positions will be taken as the origin
        :return dictionary containing the desired values as numpy arrays
        """
        offset = kwargs['offset']
        self.origin_offset = offset

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


# -------------------------------------------------------------------- TESTING -------------------------------------------------------------------- #

#
dataroot_base = '/data/sets/nuscenes'
# dataroot_train = '/media/juan/Elements'
#
# # dataset attributes
# #dataroot = dataroot_train + dataroot_base
dataroot = dataroot_base
#
# #version = 'v1.0-trainval'
version = 'v1.0-mini'
#
# #data_name = 'train'
data_name = 'mini_train'
#
nuscenes_loader = NuscenesLoader(DATAROOT=dataroot, pickle=False, version=version, data_name=data_name)
# print(nuscenes_loader.check_consistency())
#
# trajectories_dataset = nuscenes_loader.get_trajectories_data(size=20, mode='overlap', offset=10)
#
#
# # get tensorflow dataset
# pos_dataset = tf.data.Dataset.from_tensor_slices(trajectories_dataset['abs_pos'])
# rot_dataset = tf.data.Dataset.from_tensor_slices(trajectories_dataset['rotation'])
#
# full_dataset = tf.data.Dataset.zip((pos_dataset, rot_dataset))
#
# pos_element = pos_dataset.take(1)
# rot_element = rot_dataset.take(1)
# full_element = full_dataset.take(1)
#
# for i, x in enumerate(pos_element):
#     print(i, " ", x)
#
# for i, x in enumerate(rot_element):
#     print(i, " ", x)
#
# for i, (x, y) in enumerate(full_element):
#     print(tf.concat((x, y), axis=1))
#
#
# agents_list = list(nuscenes_loader.dataset['agents'].values())
#
#
# save_pkl_data(trajectories_dataset, '/data/traj_data.pkl')
# data = load_pkl_data('/data/traj_data.pkl')
# -------------------------------------------------------------------- MAPA --------------------------------------------------------------------


# This is the path where you stored your copy of the nuScenes dataset.
#DATAROOT = dataroot

# nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
# helper = PredictHelper(nuscenes)
# mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
# toks = [x.split('_') for x in mini_train]

#nusc_map = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
#fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=2)
#fig.show()


#nuscenes_loader.plotMasks(agents_list[0])

agents = list(nuscenes_loader.dataset['agents'].values())
agent = agents[0]
#agent.plotMasks(nuscenes_loader.maps)

nuscenes_loader.get_trajectories_indexes(15)

mat = agent.get_transformer_matrix(nuscenes_loader.dataset['agents'], 0)


mat