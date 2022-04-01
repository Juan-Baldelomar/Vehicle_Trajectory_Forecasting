
# base class
from dataloader import Loader
from Dataset import *

# tools
import os.path

# data manipulation and graphics
import numpy as np

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
    * self.dataset is a Dataset object that should be filled with the implementarion of load_data()

    * IMPORTANT:
        1. a SCENE is made by several moments in time or time steps. This time steps are called SAMPLES.
        2. It is relevant to know that the information of an agent (INSTANCE table) in time is stored in the SAMPLE_ANNOTATION table.
           In a database context this would be the 'MASTER' table. An agent(instance table) has several recordings in time (sample table)
           and a sample has several agents in it, so a master table is needed, therefore the sample_annotation table fullfills this.
    """

    def __init__(self, DATAROOT='/data/sets/nuscenes', pickle=True, pickle_filename='/data/sets/nuscenes/pickle/nuscenes_data.pkl',
                 version='v1.0-mini', data_name='mini_train', loadMap=True, verbose=True, rel_offset=10):

        # parent constructor
        super(NuscenesLoader, self).__init__(DATAROOT, verbose)

        # specify nuscenes attributes
        self.version: str = version
        self.data_name: str = data_name
        self.rel_offset = rel_offset
        self.nuscenes = None
        self.helper = None

        # nuscenes map expansion attributes
        if loadMap:
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
            self.nuscenes = NuScenes(version, dataroot=DATAROOT)
            self.helper = PredictHelper(self.nuscenes)
            self.get_context_information()
            self.load_ego_vehicles()
            self.dataset.agents = self.load_data()
            self.save_pickle_data(pickle_filename)

        Agent.context_dict = self.dataset.contexts

    # set verbose mode which determines if it should print the relevant information while processing the data
    def setVerbose(self, verbose: bool):
        self.verbose = verbose

    # create and insert neighbors to the context dictionary
    def insert_context_neighbor(self, instance_token: str, sample_token: str):
        self.dataset.contexts[sample_token].neighbors.add(instance_token)

    # get attributes of and agent in a specific moment in time (sample)
    def __get_agent_attributes(self, sample_annotation) -> tuple:
        # get tokens and sample
        sample_token = sample_annotation['sample_token']
        instance_token = sample_annotation['instance_token']
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
        ego_pose_x, ego_pose_y = ego_pose['translation'][:2]
        ego_rotation = ego_pose['rotation']

        return abs_pos[0], abs_pos[1], rotation, speed, accel, heading_rate, ego_pose_x, ego_pose_y, ego_rotation

    def load_ego_vehicles(self):
        scenes = self.nuscenes.scene
        for scene in scenes:
            # read token and use it as id for ego_vehicle
            token = scene['token']
            # get the map_name of the agent
            location = self.nuscenes.get('log', scene['log_token'])['location']

            self.dataset.add_ego_vehicle(token, EgoVehicle(token, location))
            # read its first sample token
            sample_token = scene['first_sample_token']

            while sample_token != '':
                sample = self.nuscenes.get('sample', sample_token)

                # get sample_data_token of a sensor to read EGO_POSE
                sample_data_token = sample['data']['RADAR_FRONT']

                # get ego_pose
                ego_pose_token = self.nuscenes.get('sample_data', sample_data_token)['ego_pose_token']
                ego_pose = self.nuscenes.get('ego_pose', ego_pose_token)
                ego_pose_x, ego_pose_y = ego_pose['translation'][:2]
                ego_rotation = ego_pose['rotation']

                self.dataset.ego_vehicles[token].add_step(sample_token, Egostep(ego_pose_x, ego_pose_y, ego_rotation))
                sample_token = sample['next']

    def load_data(self) -> dict:
        """
        this function traverses the agents in the nuscenes scheme and stores all the relevant information such as positions, rotations, etc.
        :return: dictionary object  with agents and their information
        """
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
                # get head_annotation
                first_annotation_token: str = instance['first_annotation_token']
                first_annotation: dict = self.nuscenes.get('sample_annotation', first_annotation_token)
                # tmp annotation to traverse them
                tmp_annotation = first_annotation
                # get the map_name of the agent
                scene_token = self.nuscenes.get('sample', first_annotation['sample_token'])['scene_token']
                scene = self.nuscenes.get('scene', scene_token)
                location = self.nuscenes.get('log', scene['log_token'])['location']

                # agent does not exist, create new agent
                agent = Agent(instance_token, location)
                agents[instance_token] = agent
                agent.scene_token = scene_token

                # traverse forward sample_annotations from first_annotation
                while tmp_annotation is not None:
                    sample_token = tmp_annotation['sample_token']

                    # insert neighbors to corresponding context dictionary and to ego_vehicle dictionary
                    # self.dataset['ego_vehicles'][scene_token]['timesteps'][sample_token]['neighbors'].append(instance_token)
                    self.insert_context_neighbor(instance_token, sample_token)

                    # get agent attributes tuple as: [0]-> abs_pos_x, [1]-> abs_pos_y, [2]-> rotation: list, [3]-> speed: float ,
                    # [4]-> accel: float, [5]-> heading_rate: float, [6]-> ego_pose_x, [7]-> ego_pose_y, [8]-> ego_rotation : list
                    attributes: tuple = self.__get_agent_attributes(tmp_annotation)

                    # set attributes of agent
                    agent_step = AgentTimestep(attributes[0], attributes[1], attributes[2], attributes[3], attributes[4],
                                               attributes[5], attributes[6], attributes[7], attributes[8])
                    agent.add_step(sample_token, agent_step)

                    # move to next sample_annotation if possible
                    try:
                        tmp_annotation = self.nuscenes.get('sample_annotation', tmp_annotation['next'])
                    except KeyError:
                        tmp_annotation = None

                # close while
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

        for sample in self.nuscenes.sample:
            sample_token = sample['token']

            scene_token = self.nuscenes.get('sample', sample_token)['scene_token']
            scene = self.nuscenes.get('scene', scene_token)
            location = self.nuscenes.get('log', scene['log_token'])['location']

            self.dataset.contexts[sample_token] = Context(sample_token, location)
            sample_annotation_tokens = sample['anns']

            # get context information
            for ann_token in sample_annotation_tokens:
                sample_annotation = self.nuscenes.get('sample_annotation', ann_token)

                # get the instance category
                categories = sample_annotation['category_name'].split('.')

                # add the token to the corresponding list if necessary
                if categories[0] == 'human':
                    #context['humans'].append(sample_annotation['token'])
                    pass
                elif categories[0] == ('movable_object' or 'static_object'):
                    #context['objects'].append(sample_annotation['token'])
                    pass
                elif categories[0] == 'vehicle':
                    #context['non_pred_neighbors'].add(sample_annotation['token'])
                    pass
