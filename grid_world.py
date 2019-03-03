import os 
curr_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_path)
from models import DeepDoubleSarsa, Double_Sarsa, Expected_Double_Sarsa, ReplayBuffer
import numpy as np 
import matplotlib.pyplot as plt
import random
import operator as op 
import torch
from torch.autograd import Variable


episodes = 2000
batch_size = 32
BUFFER_SIZE = 1000
epsilon_decay = 1000

    
def act(q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return random.choice([0,1,2,3])
    else:
        avg = [i+j for i,j in zip(np.squeeze(q1.cpu().data.numpy()),np.squeeze(q2.cpu().data.numpy()))]
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

def grid_world():
    env = np.arange(16)+1
    env = np.reshape(env, (4,4))
    actions = np.array(['r','l','u','d'])

    # 2 env[0,1]=A, 6 env[1,1]=P, 11 env[2,2]=W, 16 env[3,3]=G
    #action 'r': x+1, 'l': x-1, 'u': y-1, 'd': y+1

    
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    alphas = []
    for alpha in np.arange(0.1, 1.0, 0.1):
        dsarsa = Double_Sarsa(env, actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
        
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


    alpha, gamma, epsilon = 0.1, 0.99, 1.0
    # for alhpa in np.arange(0.0, 1.0, 0.1):
    ddsarsaA = DeepDoubleSarsa(64, 4)
    ddsarsaA.cuda()
    ddsarsaB = DeepDoubleSarsa(64, 4)
    ddsarsaB.cuda()
    

    rewards, lossa, lossb = [], [], []
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    for e in range(episodes):
        total_reward = 0
        agent_pose = [3,0]
        phia = np.zeros(16)
        phia[12] = 1
        input1 = np.concatenate([phia, phip, phiw, phig], axis=0)

        input = Variable(torch.from_numpy(input1))
        input = input.view(-1, 64)
        input = input.float()
        q1 = ddsarsaA(input.cuda())
        q2 = ddsarsaB(input.cuda())
        a = act(q1, q2, epsilon)
        done = False
        loss = Variable(torch.from_numpy(np.array([0.0])))
        timestep = 1
        while not done:
            r, sn, done = step(agent_pose, actions[a], step_reward='s')
            agent_pose = sn
            spose = env[sn[0], sn[1]]
            phia = np.zeros(16)
            phia[spose-1] = 1

            n_input1 = np.concatenate([phia, phip, phiw, phig], axis=0)
            n_input = torch.from_numpy(n_input1)
            n_input = Variable(n_input.view(-1, 64))
            n_input = n_input.float()

            q1n = ddsarsaA(n_input.cuda())
            q2n = ddsarsaB(n_input.cuda())

            an = act(q1n, q2n, epsilon)
            # if not timestep%100:
                # done = True
            # timestep += 1
            if done:
                replay_buffer.push(input1, a, r, n_input1, an, 1.0, np.squeeze(q1n.cpu().data.numpy()), np.squeeze(q2n.cpu().data.numpy()))
            else:
                replay_buffer.push(input1, a, r, n_input1, an, 0.0, np.squeeze(q1n.cpu().data.numpy()), np.squeeze(q2n.cpu().data.numpy()))
            total_reward +=r 

            if len(replay_buffer)>batch_size:
               
                s, a, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(batch_size)
                if np.random.rand(1)[0] > 0.5:
                    loss = ddsarsaA.update([s, a, r, sp, ap, d], q2nn, gamma)
                    lossa.append(loss.item())
                else:
                    loss = ddsarsaB.update([s, a, r, sp, ap, d], q1nn, gamma)
                    lossb.append(loss.item())
            a = an
            input = n_input
        print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.item()))
        rewards.append(total_reward)
        if e<epsilon_decay:
                epsilon -= (1.0 - 1.0)/epsilon_decay
    np.save('gws_p12', rewards)
    plt.plot(rewards)
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.title("Rewards over Episodes")
    plt.savefig('gws_p12.png')


    plt.show()


if __name__ == '__main__':
    deep_grid_world()