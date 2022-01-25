
from dataloader import Loader
from Agent import Agent
import numpy as np
from pyquaternion import Quaternion


class InputQuery:

    def __init__(self, dataloader: Loader):
        self.dataloader = dataloader

    def get_features(self, agent:Agent, pos, x_o=0, y_o=0, origin_rot=(0, 0, 0, 1)):
        x_pos = agent.abs_pos[pos][0] - x_o
        y_pos = agent.abs_pos[pos][1] - y_o
        vel = agent.speed[pos]
        acc = agent.accel[pos]
        rel_rot = Quaternion(origin_rot).inverse * Quaternion(agent.rotation[pos])
        yaw, _, _ = rel_rot.yaw_pitch_roll
        yaw = yaw / np.pi * 180

        return x_pos, y_pos, vel, acc, yaw

    def get_indexes(self, L, start=2, overlap=0):
        ego_vehicles: dict = self.dataloader.dataset['ego_vehicles']
        for ego_id, ego_dict in ego_vehicles.items():
            tmp_start = start
            timesteps = list(ego_dict['timesteps'].items())[start:]
            for s_index, (timestep_id, timestep) in enumerate(timesteps):
                if len(timestep['neighbors']) == 0:
                    tmp_start += 1
                    continue

                if tmp_start + L < len(timesteps):
                    ego_dict['indexes'].append((tmp_start, tmp_start + L))
                    break

    def get_TransformerCube_Input(self, l=14, N=25, offset=-1):
        # get indexes of the sequences
        self.get_indexes(l)
        # useful variables
        ego_vehicles: dict = self.dataloader.dataset['ego_vehicles']
        agents: dict = self.dataloader.dataset['agents']
        list_inputs = []

        for ego_id, ego_dict in ego_vehicles.items():
            inputTensor: np.ndarray = np.zeros((l, N, 5))  # (seq, neighbors, features)

            # neighbor map to store which position in the inputTensor corresponds to each neighbor
            neighbor_pos_map = {}
            available_pos = 0

            # verify if scene is eligible for sequence prediction
            if len(ego_dict['indexes']) == 0:
                continue

            # start and end of the sequence
            start, end = ego_dict['indexes'][0]
            # timesteps that will be traversed
            timesteps = list(ego_dict['timesteps'].items())[start: end]

            # origin translation and rotation axis
            origin_x, origin_y = timesteps[offset][1]['pos'] if offset != -1 else (0, 0)
            origin_rot = timesteps[offset][1]['rot'] if offset != -1 else (0, 0, 0, 1)

            for s_index, (timestep_id, timestep) in enumerate(timesteps):
                # neighbors IDs in each timestep
                neighbors = timestep['neighbors']

                for neighbor in neighbors:
                    # verify if neighbor already exists
                    if neighbor_pos_map.get(neighbor) is None:
                        # assign position in the inputTensor to the neighbor
                        neighbor_pos_map[neighbor] = available_pos
                        available_pos += 1

                    # retrive neighbor position
                    neighbor_pos = neighbor_pos_map[neighbor]

                    # not more space in the inputTensor for the neighbor, so ignore the neighbor
                    if neighbor_pos >= N:
                        continue

                    # retrieve agent
                    agent: Agent = agents[neighbor]
                    # retrieve position where information is stored
                    timestep_pos = agent.context[timestep_id]
                    # get features of the agent in this timestep
                    inputTensor[s_index, neighbor_pos, :] = self.get_features(agent, timestep_pos, origin_x, origin_y, origin_rot)

            list_inputs.append(inputTensor)

        # return inputs
        return list_inputs
