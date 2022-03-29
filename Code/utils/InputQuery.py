
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

    def get_egocentered_input(self, ego_vehicle, agents, total_seq_l, N, get_maps, seq_number=0, offset=-1):
        """
        get input scene centered in a specific ego-vehicle timestep.
        :param ego_vehicle: ego-vehicle object target
        :param agents     : dictionary of agent objects that contain agent information
        :param total_seq_l: sequence length
        :param N          : number of neighbors
        :param get_maps   : flag that indicates whether to retrieve the map of the scene or not
        :param seq_number : ego-vehicles might contain large sequences, so we might be interested in get as many scenes as possible from them.
                            self.get_indexes should have been call, if not, assert will raise an error due to len(ego_vehicle.indexes) = 0
        :param offset     : offset int that indicates the index of the timestep to use as origin. If -1 none is taken
        :return           : InputTensor with shape (sequence, neighbors, features) and InputMask (sequence, neighbors) that masks neighbors that do not appear.
        """
        assert (len(self.indexes) > 0)

        ego_id = ego_vehicle.ego_id
        start, end = self.indexes[seq_number]
        inputTensor = np.zeros((total_seq_l, N, 5))  # (seq, neighbors, features)
        inputMask = np.ones((total_seq_l, N))  # at the beginning, all neighbors have padding

        # neighbor map to store which position in the inputTensor corresponds to each neighbor
        neighbor_pos_map = {}
        available_pos = 0

        # timesteps that will be traversed, timestep[0] = key, timestep[1] = egostep object
        timesteps = list(ego_vehicle.ego_steps.items())[start: end]
        origin_timestep = timesteps[offset][1] if offset != -1 else None

        # get map in file
        if self.dataloader.maps is not None and get_maps:
            name = 'maps/' + ego_id + '_' + str(seq_number) + '.png'
            x_start = max(timesteps[offset][1].x - 100, 0)
            y_start = max(timesteps[offset][1].y - 100, 0)
            ego_vehicle.get_map(self.dataloader.maps, name, x_start, y_start, x_offset=200, y_offset=200, dpi=51.2)

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

        return inputTensor, inputMask

    def get_agentcentered_input(self, agent, agents, total_seq_l, N, get_maps, seq_number=0, offset=-1):
        """
        get input scene centered in a specific ego-vehicle timestep.
        :param agent      : agent object target
        :param agents     : dictionary of agent objects that contain agent information
        :param total_seq_l: sequence length
        :param N          : number of neighbors
        :param get_maps   : flag that indicates whether to retrieve the map of the scene or not
        :param seq_number : ego-vehicles might contain large sequences, so we might be interested in get as many scenes as possible from them.
                            self.get_indexes should have been call, if not, assert will raise an error due to len(ego_vehicle.indexes) = 0
        :param offset     : offset int that indicates the index of the timestep to use as origin. If -1 none is taken
        :return           : InputTensor with shape (sequence, neighbors, features) and InputMask (sequence, neighbors) that masks neighbors that do not appear.
        """
        assert(len(agent.index_list) > 0)

        start, end = agent.index_list[seq_number]
        # input cubes for agent
        inputTensor = np.zeros((total_seq_l, N, 5))
        inputMask = np.ones((total_seq_l, N))  # at the beginning, all neighbors have padding

        # get neighbors and timesteps of agent
        neighbors_positions = agent.get_neighbors(i)
        timesteps = list(agent.timesteps.items())[start: end]
        origin_timestep = timesteps[offset][1] if offset != -1 else None

        # GET MAPS IF ASKED
        if self.dataloader.maps is not None and get_maps:
            name = 'maps/agents/' + agent.agent_id + '_' + str(i) + '.png'
            x_start = max(timesteps[offset][1].x - 100, 0)
            y_start = max(timesteps[offset][1].y - 100, 0)
            agent.get_map(self.dataloader.maps, name, x_start, y_start, x_offset=200, y_offset=200, dpi=51.2)

        # traverse each timestep
        for s_index, (timestep_id, timestep) in enumerate(timesteps):
            # get neighbor ids in that specific timestep
            agent_neighbor_ids = self.dataloader.dataset.contexts[timestep_id].neighbors
            # traver neighbors
            for neighbor_id in agent_neighbor_ids:
                neighbor: Agent = agents[neighbor_id]
                neighbor_pos = neighbors_positions[neighbor_id]

                # check available space for neighbor
                if neighbor_pos >= N:
                    continue
                # build input
                inputTensor[s_index, neighbor_pos, :] = neighbor.get_features(timestep_id, origin_timestep)
                inputMask[s_index, neighbor_pos] = 0

        return inputTensor, inputMask

# ---------------------------------------------------------------- FUNCTIONS TO BUILD INPUTS ----------------------------------------------------------------

    def get_TransformerCube_Input(self, inp_seq_l, tar_seq_l, N, offset=-1, get_maps=False):
        # get indexes of the sequences
        self.get_indexes(inp_seq_l + tar_seq_l)
        # USEFUL VARIABLES
        ego_vehicles: dict = self.dataloader.dataset.ego_vehicles
        agents: dict = self.dataloader.dataset.agents
        list_inputs, list_agent_ids = [], []
        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        for ego_id, ego_vehicle in ego_vehicles.items():
            for i, (start, end) in enumerate(ego_vehicle.indexes):
                # get inputTensor and its mask centered in egovehicle
                inputTensor, inputMask = self.get_egocentered_input(ego_vehicle, agents, total_seq_l, N, get_maps, i, offset)
                seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded

                # split trajectories into input and target
                inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l, N)
                list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask, inputTensor))
                list_agent_ids.append(ego_id + '_' + str(i))

        # return inputs
        return list_inputs, list_agent_ids

    def get_input_ego_change(self, inp_seq_l, tar_seq_l, N, offset=-1, get_maps=False):
        self.dataloader.get_trajectories_indexes(size=inp_seq_l + tar_seq_l, mode='overlap', overlap_points=tar_seq_l)
        # useful variables
        agents: dict = self.dataloader.dataset.agents
        list_inputs, list_agent_ids = [], []
        total_seq_l = inp_seq_l + tar_seq_l

        for agent_id in agents.keys():
            agent = agents[agent_id]
            # traverse trajectories for agent
            for i, (start, end) in enumerate(agent.index_list):
                # input cubes for agent
                inputTensor, inputMask = self.get_agentcentered_input(agent, agents, total_seq_l, N, get_maps, i, offset)
                seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded

                # split inputs into past and future
                inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask,
                                                                                     inp_seq_l, tar_seq_l, N)
                list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask, inputTensor))
                list_agent_ids.append(agent_id + '_' + str(i))

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
                    timesteps = list(neighbor.timesteps.keys())[start: end]
                    origin_offset = timesteps[offset] if offset != -1 else None

                    for s_index, timestep_id in enumerate(timesteps):
                        inputTensor[s_index, :] = neighbor.get_features(timestep_id, origin_offset)

                    # split trajectories into input and target
                    inp, inp_mask, tar, tar_mask = split_single_input(inputTensor, inputMask, inp_seq_l, tar_seq_l)
                    list_inputs.append((inp, inp_mask, tar, tar_mask, inputTensor))
                    list_agent_ids.append(neighbor_id)

        # return inputs
        return list_inputs, list_agent_ids
