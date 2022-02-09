
from dataloader import Loader
from Agent import Agent
import numpy as np
from pyquaternion import Quaternion


def verifyNan(cubes):
    for cube in cubes:
        input = cube[0]
        target = cube[3]
        for face in input:
            for row in face:
                for pos, element in enumerate(row):
                    if np.isnan(element):
                        print('[WARN]: Nan values in input at pos ', pos)

        for face in target:
            for row in face:
                for pos, element in enumerate(row):
                    if np.isnan(element):
                        print('[WARN]: Nan values in target at pos ', pos)


def split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l):
    # split trajectories into input and target
    inp, tar = inputTensor[:inp_seq_l, :, :], inputTensor[inp_seq_l - 1:, :, :]
    inp_mask, tar_mask = inputMask[:inp_seq_l, :], inputMask[inp_seq_l - 1:, :]
    seq_inpMask, seq_tarMask = seq_inputMask[:inp_seq_l], seq_inputMask[inp_seq_l - 1:]

    # create padding values
    dif_seq_l = inp_seq_l - (tar_seq_l + 1)
    zeros = np.zeros((int(abs(dif_seq_l)), N, 5))
    ones_mask = np.ones((int(abs(dif_seq_l)), N))

    # add padding values to input or target. If difference == 0 no padding needed in the sequence
    if dif_seq_l > 0:
        tar = np.append(tar, zeros, axis=0)
        tar_mask = np.append(tar_mask, ones_mask, axis=0)
        seq_tarMask = np.append(seq_tarMask, np.ones(dif_seq_l), axis=0)

    elif dif_seq_l < 0:
        inp = np.append(inp, zeros, axis=0)
        inp_mask = np.append(inp_mask, ones_mask, axis=0)
        seq_inpMask = np.append(seq_inpMask, np.ones(-dif_seq_l), axis=0)

    return inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask


class InputQuery:

    def __init__(self, dataloader: Loader):
        self.dataloader = dataloader

    def get_features(self, agent: Agent, pos, x_o=0, y_o=0, origin_rot=(0, 0, 0, 1)):
        x_pos = agent.abs_pos[pos][0] - x_o
        y_pos = agent.abs_pos[pos][1] - y_o
        vel = agent.speed[pos]
        acc = agent.accel[pos]
        rel_rot = Quaternion(origin_rot).inverse * Quaternion(agent.rotation[pos])
        yaw, _, _ = rel_rot.yaw_pitch_roll
        yaw = yaw               # in radians

        # remove nans
        i = 1
        while np.isnan(vel):
            vel = agent.speed[pos + i]
            i += 1

        i = 2
        while np.isnan(acc):
            acc = agent.accel[pos + i]
            i += 1

        return x_pos, y_pos, yaw, vel, acc,

    def get_indexes(self, L, overlap=0):
        ego_vehicles: dict = self.dataloader.dataset['ego_vehicles']
        for ego_id, ego_dict in ego_vehicles.items():
            tmp_start = 0
            timesteps = list(ego_dict['timesteps'].items())
            for s_index, (timestep_id, timestep) in enumerate(timesteps):
                if len(timestep['neighbors']) == 0:
                    tmp_start += 1
                    continue

                if tmp_start + L < len(timesteps):
                    ego_dict['indexes'].append((tmp_start, tmp_start + L))
                    break

    def get_TransformerCube_Input(self, inp_seq_l, tar_seq_l, N, offset=-1):
        # get indexes of the sequences
        self.get_indexes(inp_seq_l + tar_seq_l)

        # useful variables
        ego_vehicles: dict = self.dataloader.dataset['ego_vehicles']
        agents: dict = self.dataloader.dataset['agents']
        list_inputs = []

        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        for ego_id, ego_dict in ego_vehicles.items():
            inputTensor = np.zeros((total_seq_l, N, 5))   # (seq, neighbors, features)
            inputMask = np.ones((total_seq_l, N))         # at the beginning, all neighbors have padding
            seq_inputMask = np.zeros(total_seq_l)         # at the beginning, all sequence elements are padded

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

                    # GET neighbors position in the input. Verify if neighbor already exists
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
                    # turn off mask in this position
                    inputMask[s_index, neighbor_pos] = 0

            # split trajectories into input and target
            inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l)
            list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask))

        # return inputs
        return list_inputs

    def get_input_ego_change(self, inp_seq_l, tar_seq_l, N, offset=-1):
        # useful variables
        agents: dict = self.dataloader.dataset['agents']
        list_inputs = []
        total_seq_l = inp_seq_l + tar_seq_l

        for agent_id in agents.keys():
            agent = agents[agent_id]
            neighbors_positions = agent.get_neighbors(0)

            inputTensor = np.zeros((total_seq_l, N, 5))
            inputMask = np.ones((total_seq_l, N))  # at the beginning, all neighbors have padding
            seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded

            # use a fixed origin in the agent abs positions
            x_o, y_o = agent.abs_pos[offset] if offset >= 0 else (0, 0)
            origin_rot = agent.rotation[offset] if offset != -1 else (0, 0, 0, 1)

            for s_index, (timestep_id, timestep) in enumerate(agent.context.items()):
                agent_neighbor_ids = self.dataloader.dataset['context'][timestep_id]['neighbors']

                for neighbor_id in agent_neighbor_ids:
                    neighbor: Agent = agents[neighbor_id]
                    time_pos = neighbor.context.get(timestep_id)
                    neighbor_pos = neighbors_positions[neighbor_id]

                    if neighbor_pos >= N:
                        continue

                    inputTensor[s_index, neighbor_pos, 0] = self.get_features(neighbor, time_pos, x_o, y_o, origin_rot)
                    inputMask[s_index, neighbor_pos] = 0

            inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask,
                                                                                 inp_seq_l, tar_seq_l)
            list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask))

        return list_inputs
