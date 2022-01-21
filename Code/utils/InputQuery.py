
from dataloader import Loader
from Agent import Agent
import numpy as np
from pyquaternion import Quaternion


class InputQuery:

    def __init__(self, dataloader: Loader):
        self.dataloader = dataloader

    def get_features(self, agent:Agent, pos, x_o=0, y_o=0):
        x_pos = agent.abs_pos[pos][0] - x_o
        y_pos = agent.abs_pos[pos][1] - y_o
        vel = agent.speed[pos]
        acc = agent.accel[pos]
        yaw, _, _ = Quaternion(agent.rotation[pos]).yaw_pitch_roll
        yaw = yaw / np.pi * 180

        return x_pos, y_pos, vel, acc, yaw

    def get_TransformerCube_Input(self, l=14, N=25):
        ego_vehicles: dict = self.dataloader.dataset['ego_vehicles']
        agents: dict = self.dataloader.dataset['agents']

        list_inputs = []
        start = 2

        for ego_id, ego_dict in ego_vehicles.items():
            inputTensor: np.ndarray = np.zeros((l, N, 5))  # (seq, neighbors, features)
            neighbor_pos_map = {}
            available_pos = 0

            for s_index, (timestep_id, timestep) in enumerate(list(ego_dict.items())[start: start + l]):
                neighbors = timestep['neighbors']
                for neighbor in neighbors:
                    if neighbor_pos_map.get(neighbor) is None:
                        neighbor_pos_map[neighbor] = available_pos
                        available_pos += 1

                    neighbor_pos = neighbor_pos_map[neighbor]

                    if neighbor_pos >= N:
                        continue

                    agent: Agent = agents[neighbor]

                    timestep_pos = agent.context[timestep_id]

                    inputTensor[s_index, neighbor_pos, :] = self.get_features(agent, timestep_pos)

            list_inputs.append(inputTensor)
        return list_inputs