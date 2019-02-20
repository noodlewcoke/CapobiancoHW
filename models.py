import os
curr_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_path)

import numpy as np 
import random
import operator as op
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


class Double_Sarsa:

    def __init__(self, state_space, action_space, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q1 = {'{},{}'.format(i,j):0 for i in self.state_space.reshape(-1) for j in action_space}
        self.q2 = {'{},{}'.format(i,j):0 for i in self.state_space.reshape(-1) for j in action_space}

    def update(self, sarsa):
        s, a, r, sn, an = sarsa
        
        sa = '{},{}'.format(s,a)
        sn = '{},{}'.format(sn,an)

        if np.random.rand(1)[0] > 0.5:
            self.q1[sa] = self.q1[sa] + self.alpha*(r + self.gamma*self.q2[sn] - self.q1[sa])
        else:
            self.q2[sa] = self.q2[sa] + self.alpha*(r + self.gamma*self.q1[sn] - self.q2[sa])

    def act(self, sn):
        if np.random.rand(1)[0] <= self.epsilon:
            return random.choice(self.action_space)

        else:
            d = {'{}'.format(sa):(self.q1['{},{}'.format(sn,sa)] + self.q2['{},{}'.format(sn,sa)])/2 for sa in self.action_space}
            return max(d.items(), key=op.itemgetter(1))[0]
           
    def new_alpha(self, alpha):
        self.alpha = alpha
    
    def new_gamma(self, gamma):
        self.gamma = gamma


class Expected_Double_Sarsa:
    def __init__(self, state_space, action_space, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q1 = {'{},{}'.format(i,j):0 for i in self.state_space.reshape(-1) for j in action_space}
        self.q2 = {'{},{}'.format(i,j):0 for i in self.state_space.reshape(-1) for j in action_space}

    def update(self, sarsa):
        s, a, r, sn, an = sarsa
        
        sa = '{},{}'.format(s,a)
        # sn = '{},{}'.format(sn,an)

        if np.random.rand(1)[0] > 0.5:
            d1 = {'{}'.format(a):(self.q1['{},{}'.format(sn,a)] + self.q2['{},{}'.format(sn,a)])/2 for a in self.action_space}


            self.q1[sa] = self.q1[sa] + self.alpha*(r + self.gamma*self.q2[sn] - self.q1[sa])
        else:
            self.q2[sa] = self.q2[sa] + self.alpha*(r + self.gamma*self.q1[sn] - self.q2[sa])

    def act(self, sn):
        if np.random.rand(1)[0] <= self.epsilon:
            return random.choice(self.action_space)

        else:
            d = {'{}'.format(sa):(self.q1['{},{}'.format(sn,sa)] + self.q2['{},{}'.format(sn,sa)])/2 for sa in self.action_space}
            return max(d.items(), key=op.itemgetter(1))[0]
           
    def new_alpha(self, alpha):
        self.alpha = alpha
    
    def new_gamma(self, gamma):
        self.gamma = gamma


class DeepDoubleSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False):
        super(DeepDoubleSarsa, self).__init__()

        self.qa1 = torch.nn.Linear(initus, 32, bias=bias)
        self.qa2 = torch.nn.Linear(32, exitus, bias=bias)

        self.qb1 = torch.nn.Linear(initus, 32, bias=bias)
        self.qb2 = torch.nn.Linear(32, exitus, bias=bias)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.01)


    def forward(self, input1, input2):
        qa = torch.nn.functional.relu(self.qa1(input1))
        qa = self.qa2(qa)

        qb = torch.nn.functional.relu(self.qb1(input2))
        qb = self.qb2(qb)

        return qa, qb

    def update(self, sarsa, gamma):
        s, a, r, sn, an = sarsa

        lossa, lossb = Variable(torch.zeros(1)), Variable(torch.zeros(1))
        if np.random.rand(1)[0] > 0.5:
            qa, qb = self(s, sn)
            lossa = torch.mean(torch.pow(r + gamma*qb[:,an] - qa[:,a],2)/2.0)
            lossa.backward()
            self.optimizer.step()
        else:
            qa, qb = self(sn, s)
            lossb = torch.mean(torch.pow(r + gamma*qa[:,an] - qb[:,a],2)/2.0)
            lossb.backward()
            self.optimizer.step()
        return lossa, lossb
    
    def act(self, state, epsilon):
        qa, qb = self(state, state)
        avg = [i+j for i,j in zip(qa.data.numpy(),qb.data.numpy())]
        if np.random.rand(1)[0] < epsilon:
            return random.choice([0,1,2,3])
        else:
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


def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def softmax_val(val ,x):
    val_x = np.exp(val-np.max(x))
    e_x = np.exp(x-np.max(x))
    return val_x / e_x.sum()

def normalization(x):
    return (x-np.min(x)/(np.max(x)-np.min(x)))


def grid_world():
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
    ddsarsa = DeepDoubleSarsa(64, 4)
    
    episodes = 1000
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
        a = ddsarsa.act(input, epsilon)
        done = False
        timesteps = 0
        while not done:
            r, sn, done = step(agent_pose, actions[a], step_reward='d')
            agent_pose = sn
            spose = env[sn[0], sn[1]]
            phia = np.zeros(16)
            phia[spose-1] = 1

            n_input = np.concatenate([phia, phip, phiw, phig], axis=0)
            n_input = torch.from_numpy(n_input)
            n_input = Variable(n_input.view(-1, 64))
            n_input = n_input.float()
            an = ddsarsa.act(n_input, epsilon)
            sarsa = [input, a, r, n_input, an]
            loss1, loss2 = ddsarsa.update(sarsa, gamma)
            a = an
            input = n_input
            total_reward +=r
            print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, (loss1.data[0]+loss2.data[0])/2))
            lossa.append(loss1.data); lossb.append(loss2.data)
        rewards.append(total_reward)
           
    # print(total_reward)
    plt.plot(rewards)
    plt.show()
    plt.plot(lossa)
    plt.show()
    plt.plot(lossb)
    plt.show()



if __name__ == '__main__':
    grid_world()
    # d = {'a':0.7, 'b':0.1, 'c':0.1, 'd':0.1}
    # print(softmax(normalization(list(d.values()))))
    # print(softmax_val(list(d.values())[1], list(d.values())))
    # for i in d.keys():
    #     s = softmax(d.values())
    #     d[i] = 0.9*d[i] + 0.1*0.25
    # print(d)
