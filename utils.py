import numpy as np 
import random
import operator as op
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym


def plot_reward(filepath, save=True):
    rewards = np.load(filepath)
    plt.plot(rewards)
    plt.title('Total Rewards over Episodes')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    if save:
        name = filepath.split('.')[0]
        plt.savefig(name+'.png')
    plt.show()

def plot_loss(loss_name, filepath, save=True):
    losses = np.load(filepath)
    plt.plot(losses)
    plt.title('{} over Episodes'.format(loss_name))
    plt.ylabel('Loss value')
    plt.xlabel('Episodes')
    if save:
        name = filepath.split('.')[0]
        plt.savefig(name+'.png')
    plt.show()


if __name__ == '__main__':
    gwd_p1 = np.load('grid_world/gwd_p1.npy')
    gwd_p12 = np.load('grid_world/gwd_p12.npy')
    gws_p1 = np.load('grid_world/gws_p1.npy')
    gws_p12 = np.load('grid_world/gws_p12.npy')

    plt.subplot(2,2,1)
    plt.plot(gwd_p1)
    plt.title("Deterministic Case From Position 2")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")

    plt.subplot(2,2,2)
    plt.plot(gwd_p12)
    plt.title("Deterministic Case From Position 13")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")

    plt.subplot(2,2,3)
    plt.plot(gws_p1)
    plt.title("Stochastic Case From Position 2")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")

    plt.subplot(2,2,4)
    plt.plot(gws_p12)
    plt.title("Stochastic Case From Position 13")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")

    plt.savefig("grid_world.png")
    plt.show()