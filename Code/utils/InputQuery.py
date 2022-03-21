
from dataloader import Loader
from Agent import Agent
from Dataset import *
import numpy as np
from pyquaternion import Quaternion


def verifyNan(cubes, ids):
    for num_agent, cube in enumerate(cubes):
        input = cube[0]
        target = cube[3]
        for face in input:
            for num_neighbor, row in enumerate(face):
                for pos, element in enumerate(row):
                    if np.isnan(element):
                        print('[WARN]: Nan values in input at neighbor ', num_neighbor, ' -- agent: ', ids[num_agent])

        for face in target:
            for num_neighbor, row in enumerate(face):
                for pos, element in enumerate(row):
                    if np.isnan(element):
                        print('[WARN]: Nan values in input at neighbor ', num_neighbor, ' -- agent: ', ids[num_agent])


def remove_nans(cube, p):
    cube = np.transpose(cube, (1, 0, 2))
    for j in range(len(cube)):
        nans_stack = []
        last_non_nan = 0

        for i in range(len(cube[0])):
            if np.isnan(cube[j, i, p]):
                nans_stack.insert(-1, i)
            else:
                last_non_nan = cube[j, i, p]
                while len(nans_stack) > 0:
                    index = nans_stack.pop(-1)
                    cube[j, index, p] = last_non_nan

        while len(nans_stack) > 0:
            index = nans_stack.pop(-1)
            cube[j, index, p] = last_non_nan

    cube = np.transpose(cube, (1, 0, 2))


def remove_nans_from_single(matrix, p):
    n, m = matrix.shape
    nans_stack = []
    last_non_nan = 0
    for i in range(n):
        if np.isnan(matrix[i, p]):
            nans_stack.insert(-1, i)
        else:
            last_non_nan = matrix[i, p]
            while len(nans_stack) > 0:
                index = nans_stack.pop(-1)
                matrix[index, p] = last_non_nan

        while len(nans_stack) > 0:
            index = nans_stack.pop(-1)
            matrix[index, p] = last_non_nan


def process_nans(cubes):
    for cube in cubes:
        inp = cube[0]
        tar = cube[3]
        remove_nans(inp, 3)
        remove_nans(tar, 3)
        remove_nans(inp, 4)
        remove_nans(tar, 4)


def process_single_nans(matrixes):
    for matrix in matrixes:
        inp = matrix[0]
        tar = matrix[2]
        remove_nans_from_single(inp, 3)
        remove_nans_from_single(tar, 3)
        remove_nans_from_single(inp, 4)
        remove_nans_from_single(tar, 4)


def contains_nans(matrixes):
    for matrix in matrixes:
        if np.sum(np.isnan(matrix[0])) + np.sum(np.isnan(matrix[2])) > 0:
            return True

    return False


def split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l, N):
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


def split_single_input(inputTensor, inputMask, inp_seq_l, tar_seq_l):
    # split trajectories into input and target
    inp, tar = inputTensor[:inp_seq_l, :], inputTensor[inp_seq_l - 1:, :]
    inp_mask, tar_mask = inputMask[:inp_seq_l], inputMask[inp_seq_l - 1:]

    # create padding values
    dif_seq_l = inp_seq_l - (tar_seq_l + 1)             # tar_seq_l + 1 because the extra point (last point of past)
    zeros = np.zeros((int(abs(dif_seq_l)), 5))
    ones_mask = np.ones(int(abs(dif_seq_l)))

    # add padding values to input or target. If difference == 0 no padding needed in the sequence
    if dif_seq_l > 0:
        tar = np.append(tar, zeros, axis=0)
        tar_mask = np.append(tar_mask, ones_mask, axis=0)

    elif dif_seq_l < 0:
        inp = np.append(inp, zeros, axis=0)
        inp_mask = np.append(inp_mask, ones_mask, axis=0)

    return inp, inp_mask, tar, tar_mask


class InputQuery:
    def __init__(self, dataloader: Loader):
        self.dataloader = dataloader

    # get index of start and end of a trajectory for the ego vehicle
    def get_indexes(self, L, overlap=0):
        ego_vehicles: dict = self.dataloader.dataset.ego_vehicles
        for ego_id, ego_vehicle in ego_vehicles.items():
            tmp_start = 0
            timesteps = list(ego_vehicle.ego_steps.items())
            for s_index, (timestep_id, timestep) in enumerate(timesteps):
                if len(self.dataloader.dataset.contexts[timestep_id].neighbors) == 0:
                    tmp_start += 1
                    continue

                if tmp_start + L < len(timesteps):
                    ego_vehicle.indexes.append((tmp_start, tmp_start + L))
                    break

    def get_TransformerCube_Input(self, inp_seq_l, tar_seq_l, N, offset=-1):
        # get indexes of the sequences
        self.get_indexes(inp_seq_l + tar_seq_l)
        # useful variables
        ego_vehicles: dict = self.dataloader.dataset.ego_vehicles
        agents: dict = self.dataloader.dataset.agents
        list_inputs = []
        list_agent_ids = []
        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        for ego_id, ego_vehicle in ego_vehicles.items():
            inputTensor = np.zeros((total_seq_l, N, 5))   # (seq, neighbors, features)
            inputMask = np.ones((total_seq_l, N))         # at the beginning, all neighbors have padding
            seq_inputMask = np.zeros(total_seq_l)         # at the beginning, all sequence elements are padded

            # neighbor map to store which position in the inputTensor corresponds to each neighbor
            neighbor_pos_map = {}
            available_pos = 0

            for start, end in ego_vehicle.indexes:
                # timesteps that will be traversed, timestep[0] = key, timestep[1] = egostep object
                timesteps = list(ego_vehicle.ego_steps.items())[start: end]
                origin_timestep = timesteps[offset][1] if offset != -1 else None

                for s_index, (timestep_id, timestep) in enumerate(timesteps):
                    # neighbors IDs in each timestep
                    neighbors = self.dataloader.dataset.contexts[timestep_id].neighbors

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
                        # get features of the agent in this timestep
                        inputTensor[s_index, neighbor_pos, :] = agent.get_features(timestep_id, origin_timestep)
                        # turn off mask in this position
                        inputMask[s_index, neighbor_pos] = 0

                # split trajectories into input and target
                inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l, N)
                list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask, inputTensor))
                list_agent_ids.append(ego_id)

        # return inputs
        return list_inputs, list_agent_ids

    def get_input_ego_change(self, inp_seq_l, tar_seq_l, N, offset=-1):
        self.dataloader.get_trajectories_indexes(size=inp_seq_l + tar_seq_l)
        # useful variables
        agents: dict = self.dataloader.dataset.agents
        list_inputs = []
        list_agent_ids = []
        total_seq_l = inp_seq_l + tar_seq_l

        for agent_id in agents.keys():
            agent = agents[agent_id]

            # input cubes for agent
            inputTensor = np.zeros((total_seq_l, N, 5))
            inputMask = np.ones((total_seq_l, N))  # at the beginning, all neighbors have padding
            seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded

            for i, (start, end) in enumerate(agent.index_list):
                neighbors_positions = agent.get_neighbors(i)
                timesteps = list(agent.context.items())[start: end]
                origin_timestep = timesteps[offset][1] if offset != -1 else None
                # traverse each timestep
                for s_index, (timestep_id, timestep) in enumerate(timesteps):
                    agent_neighbor_ids = self.dataloader.dataset.contexts[timestep_id].neighbors

                    for neighbor_id in agent_neighbor_ids:
                        neighbor: Agent = agents[neighbor_id]
                        neighbor_pos = neighbors_positions[neighbor_id]

                        if neighbor_pos >= N:
                            continue

                        inputTensor[s_index, neighbor_pos, :] = neighbor.get_features(timestep_id, origin_timestep)
                        inputMask[s_index, neighbor_pos] = 0

                inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask,
                                                                                     inp_seq_l, tar_seq_l, N)

                list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask, inputTensor))
                list_agent_ids.append(agent_id)

        # close for agent_id in agent.keys()

        return list_inputs, list_agent_ids

    def get_single_Input(self, inp_seq_l, tar_seq_l, offset=-1):
        # get indexes of the sequences
        self.dataloader.get_trajectories_indexes(size=inp_seq_l + tar_seq_l, mode='overlap', overlap_points=tar_seq_l)

        # useful variables
        ego_vehicles: dict = self.dataloader.dataset.ego_vehicles
        list_inputs = []
        list_agent_ids = []

        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        for ego_id, ego_vehicle in ego_vehicles.items():
            # verify if scene is eligible for sequence prediction
            neighbors_ids = ego_vehicle.get_neighbors(self.dataloader.dataset.contexts)

            for neighbor_id in neighbors_ids:
                neighbor = self.dataloader.dataset.agents[neighbor_id]
                if len(neighbor.index_list) == 0:
                    continue

                for start, end in neighbor.index_list:
                    # init inputs
                    inputTensor = np.zeros((total_seq_l, 5))
                    inputMask = np.zeros(total_seq_l)
                    # timesteps that will be traversed
                    timesteps = list(neighbor.context.keys())[start: end]
                    origin_offset = timesteps[offset] if offset != -1 else None

                    for s_index, timestep_id in enumerate(timesteps):
                        inputTensor[s_index, :] = neighbor.get_features(timestep_id, origin_offset)

                    # split trajectories into input and target
                    inp, inp_mask, tar, tar_mask = split_single_input(inputTensor, inputMask, inp_seq_l, tar_seq_l)
                    list_inputs.append((inp, inp_mask, tar, tar_mask, inputTensor))
                    list_agent_ids.append(neighbor_id)

        # return inputs
        return list_inputs, list_agent_ids
