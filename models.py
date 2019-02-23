import os
curr_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_path)

import numpy as np 
import random
import operator as op
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable




class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, next_action, done, q1n, q2n):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_action, done, q1n, q2n)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):

        # np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, next_action, done, q1n, q2n = batch[0]
        # state, action, reward, next_state, next_action, done, q1n, q2n = [state], [action], [reward], [next_state], [next_action], [done], [q1n], [q2n]

        # for b in batch[1:]:
        #     s, a, r, sn, an, d, q1, q2 = b
        #     state = np.concatenate([state, [s]])
        #     action = np.concatenate([action, [a]])
        #     reward = np.concatenate([reward, [r]])
        #     next_state = np.concatenate([next_state, [sn]])
        #     next_action = np.concatenate([next_action, [an]])
        #     done = np.concatenate([done, [d]])
        #     q1n = np.concatenate([q1n, [q1]])
        #     q2n = np.concatenate([q2n, [q2]])

        state, action, reward, next_state, next_action, done, q1n, q2n = map(np.stack, zip(*batch))
        return state, action, reward, next_state, next_action, done, q1n, q2n
    
    def __len__(self):
        return len(self.buffer)

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

        d1 = np.array([(self.q1['{},{}'.format(sn,a)] + self.q2['{},{}'.format(sn,a)])/2 for a in self.action_space])
        v = 0
        pi_sn = softmax(d1/np.mean(d1))*(1-self.epsilon) + self.epsilon*np.array([0.25, 0.25, 0.25, 0.25])
        if np.random.rand(1)[0] > 0.5:
            for i in self.action_space:
                v += pi_sn[list(self.action_space).index(i)]*self.q2['{},{}'.format(sn, i)]
            
            self.q1[sa] = self.q1[sa] + self.alpha*(r + self.gamma*v - self.q1[sa])
        else:
            for i in self.action_space:
                v += pi_sn[list(self.action_space).index(i)]*self.q1['{},{}'.format(sn, i)]
            
            self.q2[sa] = self.q2[sa] + self.alpha*(r + self.gamma*v - self.q2[sa])

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

        self.q1 = torch.nn.Linear(initus, 10, bias=bias)
        self.q2 = torch.nn.Linear(10, exitus, bias=bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00025)

    def forward(self, input):
        q = torch.nn.functional.relu(self.q1(input))
        q = self.q2(q)

        return q

    def update(self, sarsa, q2, gamma):   
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s))
        sn = torch.FloatTensor(sn)
        # a = torch.FloatTensor(a)
        # an = torch.FloatTensor(an)
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))

        q = self(s)
        # print(qb)
        self.optimizer.zero_grad()
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[:,an] - q[:,a],2)/2.0)
        loss.backward()
        self.optimizer.step()
        return loss

class cnnDeepDoubleSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False):
        super(DeepDoubleSarsa, self).__init__()

        # dobbiamo usare convolutional?????
        self.cn1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        self.cn2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.cn3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = torch.nn.Linear(5760, 512, bias=bias)
        self.fc2 = torch.nn.Linear(512, exitus, bias=bias)

        ''' torch.optim is a package implementing various optimization algorithms '''
        # Adam optimizer is one of the most popular gradient descent optimizer in deep learning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, input):
        q = torch.nn.functional.relu(self.cn1(input))
        q = torch.nn.functional.relu(self.cn2(q))
        q = torch.nn.functional.relu(self.cn3(q))
        q = q.view(-1, self.num_flat_features(q))
   
        ''' Activation function (to get the output of node) : ReLU function returns a 0 for input values less than 0,
         while input values above 0, the function returns a value between 0 and 1 '''
        
        q = torch.nn.functional.relu(self.fc1(q))
        q = self.fc2(q)

        return q

    def update(self, sarsa, q2, gamma):
        s, a, r, sn, an = sarsa
        q = self(s)
        '''In PyTorch we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates 
        the gradients on subsequent backward passes'''
        self.optimizer.zero_grad()
        '''Update rule for DDS'''
        loss = torch.mean(torch.pow(r + gamma*q2[:,an] - q[:,a],2)/2.0)
        ''' Backpropagation: loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. 
        These are accumulated into x.grad for every parameter x '''
        loss.backward()
        ''' with update the weights '''
        self.optimizer.step()
        return loss
      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def softmax_val(val ,x):
    val_x = np.exp(val-np.max(x))
    e_x = np.exp(x-np.max(x))
    return val_x / e_x.sum()

def normalization(x):
    return (x-np.min(x)/(np.max(x)-np.min(x)))

if __name__ == '__main__':
    pass