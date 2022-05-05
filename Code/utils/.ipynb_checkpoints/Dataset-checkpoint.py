
class AgentTimestep:
    def __init__(self, x, y, rot, speed, accel, heading_rate, ego_pos_x, ego_pos_y, ego_rot):
        self.x = x
        self.y = y
        self.rot = rot
        self.speed = speed
        self.accel = accel
        self.heading_rate = heading_rate
        self.ego_pos_x = ego_pos_x
        self.ego_pos_y = ego_pos_y
        self.ego_rot = ego_rot


class Egostep:
    def __init__(self, x, y, rot):
        self.x = x
        self.y = y
        self.rot = rot


class Context:
    def __init__(self, context_id, location=None):
        self.context_id = context_id
        self.neighbors = set()
        self.non_pred_neighbors = {}
        self.humans = {}
        self.objects = {}
        self.map = location

    def add_non_pred_neighbor(self, agent_id, agent_timestep):
        if self.non_pred_neighbors.get(agent_id) is None:
            self.non_pred_neighbors[agent_id] = agent_timestep


class EgoVehicle:
    def __init__(self, ego_id):
        self.ego_id = ego_id
        self.indexes = []
        self.ego_steps = {}             # dict of steps in time

    def add_step(self, step_id, ego_step):
        if self.ego_steps.get(step_id) is None:
            self.ego_steps[step_id] = ego_step

    def get_neighbors(self, context_pool):
        neighbors_across_time = set()
        for timestep_id, ego_step in self.ego_steps.items():
            for neighbor in context_pool[timestep_id].neighbors:
                neighbors_across_time.add(neighbor)

        return neighbors_across_time


class Dataset:
    def __init__(self):
        self.agents = {}
        self.contexts: Context = {}
        self.ego_vehicles: EgoVehicle = {}

    def add_agent(self, agent_id, agent):
        if self.agents.get(agent_id) is None:
            self.agents[agent_id] = agent
        else:
            print("[WARN]: trying to add agent again")

    def add_context(self, context_id):
        if self.contexts.get(context_id) is None:
            self.contexts[context_id] = Context(context_id)

    def add_ego_vehicle(self, ego_id, egovehicle):
        if self.ego_vehicles.get(ego_id) is None:
            self.ego_vehicles[ego_id] = egovehicle
