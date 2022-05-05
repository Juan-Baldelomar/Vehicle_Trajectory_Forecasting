
from dataloader import Loader
from DataModel import *
import numpy as np
from pyquaternion import Quaternion


"""
This file is used to query data from the dataloader and perform several operations over the dataset. For example you can retrieve inputs in the form of the 
ego vehicle fixed in some timestep as the origin and retrieve all the positions of the neighbors in relation to this origin.
"""


def verifyNan(cubes, ids):
    for num_agent, cube in enumerate(cubes):
        input_ = cube[0]
        target = cube[3]
        for face in input_:
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


def stamp_positions_in_bitmap(inputs: np.ndarray, masks: np.ndarray, bitmaps: np.ndarray, pixbymeter: float, yaw, step_start=-1, step_end=1):
    """
    :param: inputs    : np array of the form (S, N, F). S=sequence, N=Neighbors, F=Features (x=0, y=1).
    :param: masks     : np array of the form (S, N). S=sequence, N=Neighbors. Indicates thoses entries that are padded.
    :param: bitmaps   : np array of the form (L, H, W). L=Layers, H=Height, W=Width.
    :param: pixbymeter: relation of number of pixels by each meter
    :param: yaw       : rotation angle of the points (if map was rotated, trajectories should be rotated by the same angle)
    :param: step start: when stamping the step of and agent, a pixel width might be to small, so you can make it bigger by setting the step_start
                        and step_end, so you stamp all the pixels in the range (pos + step_start, pos + step_end)
    """
    assert(step_start <= 0 and step_end >= 0)
    # needed shapes
    S, N, _ = inputs.shape
    n_layers, H, W = bitmaps.shape
    neigh_bitmaps = np.ones([N, n_layers, H, W]) * bitmaps[np.newaxis, :, :, :]
    # center pixels
    x_p, y_p = H / 2., W / 2.
    # get x, y for each observation, select the points that are not padded (masks==0) and get array flattened
    x = inputs[:, :, 0][masks == 0]
    y = inputs[:, :, 1][masks == 0]
    # perform rotation (clockwise)
    pix_x = x * np.cos(yaw) + y * np.sin(yaw)
    pix_y = -x * np.sin(yaw) + y * np.cos(yaw)
    # transform to pixel position
    pix_x = (pix_x * pixbymeter + x_p).astype(np.int32)
    pix_y = (pix_y * pixbymeter + y_p).astype(np.int32)
    pix_x = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_x))
    pix_y = np.maximum(0 - step_start, np.minimum(255 - step_end, pix_y))
    # get neighbor dimension indexes, select the ones that are not padded and get array flattened
    N_pos = np.arange(N)[np.newaxis, :] * np.ones([S, N]).astype(np.int32)
    N_pos = N_pos[masks == 0]
    # build positions layers
    stamped_positions = np.zeros((N, H, W))
    # stamp values
    for i in range(step_start, step_end + 1):
        for j in range(step_start, step_end + 1):
            stamped_positions[(N_pos, pix_y + i, pix_x + j)] = 255.0
    # append new layer
    neigh_bitmaps = np.append(neigh_bitmaps, stamped_positions[:, np.newaxis, :, :], axis=1)

    return neigh_bitmaps


class InputQuery:
    def __init__(self, dataloader: Loader):
        self.dataset = dataloader.dataset

    def get_egocentered_input(self, agent: Agent, agents, total_seq_l: int, N: int, seq_number=0,
                              offset=-1, bitmap_extractor: BitmapFeature = None, **kwargs):
        """
        get input scene centered in a specific ego-vehicle timestep.
        :param agent                :  agent object target, treated as center or virtual ego vehicle (meaning it could or could not be a real ego-vehicle)
        :param agents               : dictionary of agent objects that contain agent information
        :param total_seq_l          : sequence length
        :param N                    : number of neighbors
        :param bitmap_extractor     : mode to get maps, default None. values = masks, semantic
        :param seq_number           : ego-vehicles might contain large sequences, so we might be interested in get as many scenes as possible from them.
                                      self.get_indexes should have been call, if not, assert will raise an error due to len(ego_vehicle.indexes) = 0
        :param offset               : offset int that indicates the index of the timestep to use as origin. If -1 none is taken
        :return                     : InputTensor with shape (sequence, neighbors, features) and InputMask (sequence, neighbors) that masks neighbors that do not appear.
        """
        assert (len(agent.indexes) > 0)
        start, end = agent.indexes[seq_number]
        inputTensor = np.zeros((total_seq_l, N, 5))     # (seq, neighbors, features)
        inputMask = np.ones((total_seq_l, N))           # at the beginning, all neighbors have padding
        bitmaps = None
        # timesteps that will be traversed, timestep[0] = key, timestep[1] = egostep object
        timesteps = list(agent.timesteps.items())[start: end]
        origin_timestep: AgentTimestep = timesteps[offset][1] if offset != -1 else None

        # get bitmaps for center agent (ego vehicle or agent treated as ego vehicle)
        if bitmap_extractor is not None:
            bitmaps = bitmap_extractor.getMasks(origin_timestep, agent.map_name, **kwargs)

        # available positions start from 1 because ego vehicle occupies position 0.
        neighbors_positions = self.dataset.get_agent_neighbors(agent, seq_number)

        for s_index, (timestep_id, timestep) in enumerate(timesteps):
            # ADD EGO VEHICLE TIMESTEP AS INPUT
            ego_step = self.dataset.ego_vehicles[agent.ego_id].timesteps[timestep_id]
            inputTensor[s_index, 0, :2] = ego_step.x - origin_timestep.x, ego_step.y - origin_timestep.y
            inputTensor[s_index, 0,  2] = ego_step.rot - origin_timestep.rot
            inputMask[s_index, 0] = 0

            # neighbors IDs in each timestep
            neighbors = self.dataset.contexts[timestep_id].neighbors

            for neighbor in neighbors:
                # retrive neighbor position
                neighbor_pos = neighbors_positions[neighbor]
                # not more space in the inputTensor for the neighbor, so ignore the neighbor
                if neighbor_pos >= N:
                    continue
                # retrieve agent
                agent: NuscenesAgent = agents[neighbor]
                # get features of the agent in this timestep
                inputTensor[s_index, neighbor_pos, :] = agent.get_features(timestep_id, origin_timestep)
                # turn off mask in this position
                inputMask[s_index, neighbor_pos] = 0

        return inputTensor, inputMask, bitmaps, origin_timestep.rot

# ---------------------------------------------------------------- FUNCTIONS TO BUILD INPUTS ----------------------------------------------------------------

    def get_TransformerCube_Input(self, inp_seq_l, tar_seq_l, N, offset=-1, use_ego_vehicles=True,
                                  bitmap_extractor: BitmapFeature = None, path='maps', **kwargs):
        # get indexes of the sequences
        self.dataset.get_trajectories_indexes(use_ego_vehicles=use_ego_vehicles, L=inp_seq_l + tar_seq_l)
        # USEFUL VARIABLES
        ego_vehicles: dict = self.dataset.ego_vehicles if use_ego_vehicles else self.dataset.agents
        agents: dict = self.dataset.agents
        list_inputs = []
        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        # traverse all ego vehicles
        for ego_id, ego_vehicle in ego_vehicles.items():
            # traverse all the possible trajectories for and ego vehicle
            for i, (_, _) in enumerate(ego_vehicle.indexes):
                # get inputTensor and its mask centered in egovehicle
                inputTensor, inputMask, bitmaps, yaw = self.get_egocentered_input(ego_vehicle, agents, total_seq_l, N, seq_number=i,
                                                                                  offset=offset, bitmap_extractor=bitmap_extractor, **kwargs)
                seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded
                # split trajectories into input and target
                name = ego_id + '_' + str(i)
                inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l, N)
                list_inputs.append({'past': inp,
                                    'past_neighMask': inp_mask,
                                    'past_seqMask': seq_inpMask,
                                    'future': tar,
                                    'future_neighMask': tar_mask,
                                    'future_seqMask': seq_tarMask,
                                    'full_traj': inputTensor,
                                    'origin_yaw': yaw,
                                    'ego_id': name})
                # save bitmaps and store name
                if bitmap_extractor is not None:
                    np.savez_compressed('/'.join([path, name]), bitmaps=bitmaps)

        # return inputs
        return list_inputs

    def get_single_Input(self, inp_seq_l, tar_seq_l, offset=-1):
        # get indexes of the sequences
        self.dataset.get_trajectories_indexes(size=inp_seq_l + tar_seq_l, mode='overlap', overlap_points=tar_seq_l)

        # useful variables
        ego_vehicles: dict = self.dataset.ego_vehicles
        list_inputs = []
        list_agent_ids = []

        # max sequence lenght
        total_seq_l = inp_seq_l + tar_seq_l

        for ego_id, ego_vehicle in ego_vehicles.items():
            # verify if scene is eligible for sequence prediction
            neighbors_ids = ego_vehicle.get_neighbors(self.dataset.contexts)

            for neighbor_id in neighbors_ids:
                neighbor = self.dataset.agents[neighbor_id]
                if len(neighbor.indexes) == 0:
                    continue

                for start, end in neighbor.indexes:
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
