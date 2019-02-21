import os 
curr_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_path)
from models import DeepDoubleSarsa, Double_Sarsa, Expected_Double_Sarsa
import numpy as np 
import matplotlib.pyplot as plt
import random
import operator as op 
import gym
import torch
from torch.autograd import Variable


def grid_world():
    env = np.arange(16)+1
    env = np.reshape(env, (4,4))
    actions = np.array(['r','l','u','d'])
    # 2 env[0,1]=A, 6 env[1,1]=P, 11 env[2,2]=W, 16 env[3,3]=G
    #action 'r': x+1, 'l': x-1, 'u': y-1, 'd': y+1

    
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    alphas = []
    for alpha in np.arange(0.1, 1.0, 0.1):
        dsarsa = Expected_Double_Sarsa(env, actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        episodes = 10000
        rewards = []
        for e in range(episodes):
            total_reward = 0
            agent_pose = [0,1]
            s = 2
            a = dsarsa.act(s)
            done = False
            while not done:
                r, sn, done = step(agent_pose, a, step_reward='d')
                agent_pose = sn
                sn = env[sn[0], sn[1]]
                an = dsarsa.act(sn)
                sarsa = [s, a, r, sn, an]
                dsarsa.update(sarsa)
                # dsarsa.new_alpha(dsarsa.alpha+1/episodes)
                a = an
                s = sn
                total_reward +=r
                print("Alpha: {} | Episode: {} | Reward: {}".format(alpha, e, total_reward))
                # epsilon -= 0.5/episodes
                
            rewards.append(total_reward)
        alphas.append(np.mean(rewards))

    plt.plot(np.arange(0.1, 1.0, 0.1), alphas)
    plt.ylabel("Rewards")
    plt.xlabel("alpha")
    plt.savefig('edsarsa_dgrid.png')
    plt.show()


def deep_grid_world():
    env = np.arange(16)+1
    env = np.reshape(env, (4,4))
    actions = np.array(['r','l','u','d'])
    # 2 env[0,1]=A, 6 env[1,1]=P, 11 env[2,2]=W, 16 env[3,3]=G
    #action 'r': x+1, 'l': x-1, 'u': y-1, 'd': y+1
    phip = np.zeros(16)
    phip[5] = 1
    phiw = np.zeros(16)
    phiw[10] = 1
    phig = np.zeros(16)
    phig[15] = 1


    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    # for alhpa in np.arange(0.0, 1.0, 0.1):
    ddsarsaA = DeepDoubleSarsa(64, 4)
    ddsarsaB = DeepDoubleSarsa(64, 4)
    

    episodes = 50000
    rewards, lossa, lossb = [], [], []
    for e in range(episodes):
        total_reward = 0
        agent_pose = [0,1]
        phia = np.zeros(16)
        phia[1] = 1
        input = np.concatenate([phia, phip, phiw, phig], axis=0)
        # print(input.shape)
        # input = torch.from_numpy(input)
        input = Variable(torch.from_numpy(input))
        input = input.view(-1, 64)
        input = input.float()
        # print(input)
        # exit()
        q1 = ddsarsaA(input)
        q2 = ddsarsaB(input)
        a = act(q1, q2, epsilon)
        done = False
        timesteps = 0
        while not done:
            r, sn, done = step(agent_pose, actions[a], step_reward=None)
            agent_pose = sn
            spose = env[sn[0], sn[1]]
            phia = np.zeros(16)
            phia[spose-1] = 1

            n_input = np.concatenate([phia, phip, phiw, phig], axis=0)
            n_input = torch.from_numpy(n_input)
            n_input = Variable(n_input.view(-1, 64))
            n_input = n_input.float()

            q1n = ddsarsaA(n_input)
            q2n = ddsarsaB(n_input)

            an = act(q1n, q2n, epsilon)
            sarsa = [input, a, r, n_input, an]
            loss = 0
            if np.random.rand(1)[0] > 0.5:
                loss = ddsarsaA.update(sarsa, q2n, gamma)
                lossa.append(loss.data[0])
            else:
                loss = ddsarsaB.update(sarsa, q1n, gamma)
                lossb.append(loss.data[0])
            a = an
            input = n_input
            total_reward +=r
            print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.data[0]))
        rewards.append(total_reward)
           
    # print(total_reward)
    plt.subplot(3,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3,1,2)
    plt.plot(lossa)
    plt.title("Lossa")

    plt.subplot(3,1,3)
    plt.plot(lossb)
    plt.title("Lossb")

    plt.show()


def gym_cartpole():
    env = gym.make('CartPole-v1')

    obs_len = env.observation_space.shape[0]
    
    ddsarsa1 = DeepDoubleSarsa(obs_len, 2)
    ddsarsa2 = DeepDoubleSarsa(obs_len, 2)


    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    episodes = 10000
    rewards, lossa, lossb = [], [], []
    for e in range(episodes):
        done = False
        total_reward = 0

        obs = env.reset()
        obs = Variable(torch.from_numpy(obs))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        while not done:

            # env.render()
            qa = ddsarsa1(obs)
            qb = ddsarsa2(obs)
            a = gym_act(env, qa, qb, epsilon)
            n_obs, r, done, _ = env.step(a)

            n_obs = Variable(torch.from_numpy(n_obs))
            n_obs = n_obs.view(-1, obs_len)
            n_obs = n_obs.float()
            n_qa = ddsarsa1(n_obs)
            n_qb = ddsarsa2(n_obs)
            an = gym_act(env, n_qa, n_qb, epsilon)
            sarsa = [obs, a, r, n_obs, an]
            loss = 0
            if np.random.rand(1)[0] > 0.5:
                loss = ddsarsa1.update(sarsa, n_qb, gamma)
                lossa.append(loss.data[0])
            else:
                loss = ddsarsa2.update(sarsa, n_qa, gamma)
                lossb.append(loss.data[0])
            obs = n_obs
            a = an
            total_reward +=r
            print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.data[0]))
        rewards.append(total_reward)
           
    # print(total_reward)
    plt.subplot(3,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3,1,2)
    plt.plot(lossa)
    plt.title("Lossa")

    plt.subplot(3,1,3)
    plt.plot(lossb)
    plt.title("Lossb")

    plt.savefig('cartpole/dd_10.png')
    plt.show()


def gym_act(env, q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return env.action_space.sample()
    else:
        avg = [i+j for i,j in zip(q1.data.numpy(),q2.data.numpy())]
        return np.argmax(np.array(avg))
    
def act(q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return random.choice([0,1,2,3])
    else:
        avg = [i+j for i,j in zip(q1.data.numpy(),q2.data.numpy())]
        return np.argmax(np.array(avg))


def step(state, action, step_reward=None):
    r, sn, done = 0, None, False 
    if action=='r' and not state[1]==3 and not [state[0], state[1]+1]==[2,2]:
        sn = [state[0], state[1]+1]
    elif action=='l' and not state[1]==0 and not [state[0], state[1]-1]==[2,2]:
        sn = [state[0], state[1]-1]
    elif action=='u' and not state[0]==0 and not [state[0]-1, state[1]]==[2,2]:
        sn = [state[0]-1, state[1]]
    elif action=='d' and not state[0]==3 and not [state[0]+1, state[1]]==[2,2]:
        sn = [state[0]+1, state[1]]
    else:
        sn = state

    if sn==[1,1]:
        r = -10
        done = True
    elif sn==[3,3]:
        r = 10
        done = True
    else:
        if step_reward=='d':
            r = np.random.normal(loc=-1, scale=0)
        elif step_reward=='s':
            if np.random.rand(1)[0] > 0.5:
                r = -8
            else:
                r = 6
    return r, sn, done


if __name__ == '__main__':
    # grid_world()
    gym_cartpole()
    
    pass


