import os 
curr_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_path)
# from models import DeepDoubleSarsa, Double_Sarsa, Expected_Double_Sarsa, ReplayBuffer
import numpy as np 
import matplotlib.pyplot as plt
import random
import operator as op 
import gym
import torch
from torch.autograd import Variable
from deepmind_training import DeepDoubleSarsa as dds 
from skimage import color
import cv2


def gym_cartpole():
    env = gym.make('MountainCar-v0')

    obs_len = env.observation_space.shape[0]
    
    ddsarsa1 = DeepDoubleSarsa(obs_len, 3)
    ddsarsa2 = DeepDoubleSarsa(obs_len, 3)


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
        avg = [i+j for i,j in zip(np.squeeze(q1.cpu().data.numpy()),np.squeeze(q2.cpu().data.numpy()))]
        return np.argmax(np.array(avg))
    
def act(q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return random.choice([0,1,2,3])
    else:
        avg = [300*(i+j) for i,j in zip(q1.data.numpy(),q2.data.numpy())]
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

class DeepDoubleSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False):
        super(DeepDoubleSarsa, self).__init__()

        # dobbiamo usare convolutional?????
        self.cn1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        self.cn1b = torch.nn.BatchNorm2d(32)
        self.cn2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.cn2b = torch.nn.BatchNorm2d(64)
        self.cn3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.cn3b = torch.nn.BatchNorm2d(64)
        
        self.fc1 = torch.nn.Linear(5760, 512, bias=bias)
        # self.fc2 = torch.nn.Linear(64, 64, bias=bias)

        # self.fc3 = torch.nn.Linear(64, 64, bias=bias)
        # self.fc1b = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, exitus, bias=bias)
        
        

        ''' torch.optim is a package implementing various optimization algorithms '''
        # Adam optimizer is one of the most popular gradient descent optimizer in deep learning
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)

    def forward(self, input):
        q = torch.nn.functional.relu(self.cn1b(self.cn1(input)))
        q = torch.nn.functional.relu(self.cn2b(self.cn2(q)))
        q = torch.nn.functional.relu(self.cn3b(self.cn3(q)))
        # q = q.view(-1, self.num_flat_features(q))

        ''' Activation function (to get the output of node) : ReLU function returns a 0 for input values less than 0,
         while input values above 0, the function returns a value between 0 and 1 '''
        # q = input
        q = q.view(-1, self.num_flat_features(q))
        q = torch.nn.functional.relu(self.fc1(q))
        # q = torch.nn.functional.relu(self.fc2(q))
        q = self.fc2(q)

        return q

    def update(self, sarsa, q2, gamma):
     
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))
        
        q = self(s.view(-1, 1, 84, 80))

        in_q = [np.arange(len(a)), a]
        in_qb = [np.arange(len(an)), an]
        '''In PyTorch we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates 
        the gradients on subsequent backward passes'''
        self.optimizer.zero_grad()
        '''Update rule for DDS'''
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[in_qb] - q.cpu()[in_q],2)/2.0)
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
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))


def gym_bowling():
    env = gym.make('Bowling-v0')

    obs_len = env.observation_space.shape[0]
    
    ddsarsa1 = dds(obs_len, 6, bias=True)
    ddsarsa1.to("cuda")
    ddsarsa2 = dds(obs_len, 6, bias=True)
    ddsarsa2.to("cuda")
    ddsarsa1.load('models/bowling_dm3a.pt')
    ddsarsa2.load('models/bowling_dm3b.pt')


    alpha, gamma, epsilon = 0.1, 0.99, 0.0
    episodes = 10000
    rewards, lossa, lossb = [], [], []
    for e in range(episodes):
        done = False
        total_reward = 0

        obs = env.reset()
        obs = color.rgb2gray(obs)
        obs = cv2.resize(obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
        obs = Variable(torch.from_numpy(obs))
        obs = obs.view(-1, 1, 84, 80)
        obs = obs.float()
        obs = obs.to("cuda")
        while not done:

            # env.render()
            qa = ddsarsa1(obs)
            qb = ddsarsa2(obs)
            # print(np.squeeze(qa.cpu().data.numpy()))
            # print(np.squeeze(qb.cpu().data.numpy()))

            a = gym_act(env, qa, qb, epsilon)
            n_obs, r, done, _ = env.step(a)
            
            n_obs = color.rgb2gray(n_obs)
            n_obs = cv2.resize(n_obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
            n_obs = Variable(torch.from_numpy(n_obs))
            n_obs = n_obs.view(-1, 1, 84, 80)
            n_obs = n_obs.float()
            n_obs = n_obs.to("cuda")
            obs = n_obs
            total_reward +=r
        print("Episode: {} | Reward: {}".format(e, total_reward))
        rewards.append(total_reward)

    plt.subplot(3,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3,1,2)
    plt.plot(lossa)
    plt.title("Lossa")

    plt.subplot(3,1,3)
    plt.plot(lossb)
    plt.title("Lossb")

    plt.savefig('dd_10.png')
    plt.show()

def test_act(q):
    return np.argmax(np.squeeze(q.cpu().data.numpy()))


def test_cartpole():
    env = gym.make('CartPole-v1')
    obs_len = env.observation_space.shape[0]
    
    ddsarsa1 = dds(obs_len, 2, bias=True)
    ddsarsa1.to("cuda")
    ddsarsa2 = dds(obs_len, 2, bias=True)
    ddsarsa2.to("cuda")
    ddsarsa1.load('models/cartpole/cartpole_dm1a.pt')
    ddsarsa2.load('models/cartpole/cartpole_dm1b.pt')


    alpha, gamma, epsilon = 0.1, 0.99, 0.0
    episodes = 100
    rewards, lossa, lossb = [], [], []
    for e in range(episodes):
        done = False
        total_reward = 0

        obs = env.reset()
        obs = Variable(torch.from_numpy(obs))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        obs = obs.to("cuda")
        while not done:

            env.render()
            qa = ddsarsa1(obs)
            qb = ddsarsa2(obs)
            # print(np.squeeze(qa.cpu().data.numpy()))
            # print(np.squeeze(qb.cpu().data.numpy()))

            # a = gym_act(env, qa, qb, epsilon)
            a = test_act(qb)
            n_obs, r, done, _ = env.step(a)
            
            n_obs = Variable(torch.from_numpy(n_obs))
            n_obs = n_obs.view(-1, obs_len)
            n_obs = n_obs.float()
            n_obs = n_obs.to("cuda")
            obs = n_obs
            total_reward +=r
        print("Episode: {} | Reward: {}".format(e, total_reward))
        rewards.append(total_reward)
    env.close()

if __name__ == '__main__':
    # deep_grid_world()
    # gym_cartpole()
    # gym_bowling()
    test_cartpole()
    pass


