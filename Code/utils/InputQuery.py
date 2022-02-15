
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


def process_nans(cubes):
    for cube in cubes:
        inp = cube[0]
        tar = cube[3]
        remove_nans(inp, 3)
        remove_nans(tar, 3)
        remove_nans(inp, 4)
        remove_nans(tar, 4)


# get input features of an agent
def get_features(agent: Agent, step_id, x_o=0, y_o=0, origin_rot=(0, 0, 0, 1)):
    agent_time_step : AgentTimestep = agent.context[step_id]
    x_pos = agent_time_step.x - x_o
    y_pos = agent_time_step.y - y_o
    vel = agent_time_step.speed
    acc = agent_time_step.accel
    rel_rot = Quaternion(origin_rot).inverse * Quaternion(agent_time_step.rot)
    yaw, _, _ = rel_rot.yaw_pitch_roll
    yaw = yaw               # in radians
    return x_pos, y_pos, yaw, vel, acc


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

            # verify if scene is eligible for sequence prediction
            if len(ego_vehicle.indexes) == 0:
                continue

            # start and end of the sequence
            start, end = ego_vehicle.indexes[0]
            # timesteps that will be traversed
            timesteps: list[tuple[str, Egostep]] = list(ego_vehicle.ego_steps.items())[start: end]

            # origin translation and rotation axis
            origin_x, origin_y = timesteps[offset][1].x, timesteps[offset][1].y if offset != -1 else (0, 0)
            origin_rot = timesteps[offset][1].rot if offset != -1 else (0, 0, 0, 1)

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
                    inputTensor[s_index, neighbor_pos, :] = get_features(agent, timestep_id, origin_x, origin_y, origin_rot)
                    # turn off mask in this position
                    inputMask[s_index, neighbor_pos] = 0

            # split trajectories into input and target
            inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask, inp_seq_l, tar_seq_l, N)
            list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask))
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
            neighbors_positions = agent.get_neighbors(0)

            # input cubes for agent
            inputTensor = np.zeros((total_seq_l, N, 5))
            inputMask = np.ones((total_seq_l, N))  # at the beginning, all neighbors have padding
            seq_inputMask = np.zeros(total_seq_l)  # at the beginning, all sequence elements are padded

            # trajectory with no sufficient points
            if len(agent.index_list) == 0:
                continue

            # get start and end of trajectory and get its timesteps
            start, end = agent.index_list[0]
            timesteps: list[tuple[str, AgentTimestep]] = list(agent.context.items())[start: end]

            # use a fixed origin in the agent abs positions
            x_o, y_o = timesteps[offset][1].x, timesteps[offset][1].y if offset >= 0 else (0, 0)
            origin_rot = timesteps[offset][1].rot if offset != -1 else (0, 0, 0, 1)

            # traverse each timestep
            for s_index, (timestep_id, timestep) in enumerate(timesteps):
                agent_neighbor_ids = self.dataloader.dataset.contexts[timestep_id].neighbors

                for neighbor_id in agent_neighbor_ids:
                    neighbor: Agent = agents[neighbor_id]
                    neighbor_pos = neighbors_positions[neighbor_id]

                    if neighbor_pos >= N:
                        continue

                    inputTensor[s_index, neighbor_pos, :] = get_features(neighbor, timestep_id, x_o, y_o, origin_rot)
                    inputMask[s_index, neighbor_pos] = 0

            inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask = split_input(inputTensor, inputMask, seq_inputMask,
                                                                                 inp_seq_l, tar_seq_l, N)
            list_inputs.append((inp, inp_mask, seq_inpMask, tar, tar_mask, seq_tarMask))
            list_agent_ids.append(agent_id)

        # close for agent_id in agent.keys()

        return list_inputs, list_agent_ids
