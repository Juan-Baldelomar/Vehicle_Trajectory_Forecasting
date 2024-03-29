
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches


def ADE(real, pred):
    diff_sq = (real - pred)**2
    diff_sq = np.sum(diff_sq, axis=2)
    diff_sq = np.sqrt(diff_sq)
    mean_diff = np.mean(diff_sq)
    return mean_diff

def FDE(real, pred):
    diff_sq = (real[:, -1:, :] - pred[:, -1:, :])**2
    diff_sq = np.sum(diff_sq, axis=2)
    diff_sq = np.sqrt(diff_sq)
    mean_diff = np.mean(diff_sq)
    return mean_diff

#ADE(real, pred)


#def plot_traj(agent_traj):
#    x = agent_traj[:, 0]
#    y = agent_traj[:, 1]
#    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
#    axs.scatter(x, y)
#    for step in agent_traj:
#        x, y, yaw = step[0], step[1], step[2] * 180 / np.pi
#        rect = mpatches.Rectangle((x, y), 1.5 / 10, 2.5 / 10, angle=-yaw, fill=True, color="purple", linewidth=1)
#        fig.gca().add_patch(rect)
#
#    fig.show()

