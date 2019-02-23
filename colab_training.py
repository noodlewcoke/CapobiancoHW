import numpy as np 
import random
import operator as op
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym
from skimage import color
import cv2
from google.colab import files

LOAD = False
SAVE = True
saving_fq = 2


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
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_action, done, q1n, q2n = map(np.stack, zip(*batch))

        return state, action, reward, next_state, next_action, done, q1n, q2n
    
    def __len__(self):
        return len(self.buffer)
      
      
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
        self.fc1b = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, exitus, bias=bias)
        

        ''' torch.optim is a package implementing various optimization algorithms '''
        # Adam optimizer is one of the most popular gradient descent optimizer in deep learning
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)

    def forward(self, input):
        q = torch.nn.functional.relu(self.cn1b(self.cn1(input)))
        #print(q.shape)
        q = torch.nn.functional.relu(self.cn2b(self.cn2(q)))
        q = torch.nn.functional.relu(self.cn3b(self.cn3(q)))
        q = q.view(-1, self.num_flat_features(q))
        #print(q.shape)
        ''' Activation function (to get the output of node) : ReLU function returns a 0 for input values less than 0,
         while input values above 0, the function returns a value between 0 and 1 '''
        
        q = torch.nn.functional.relu(self.fc1(q))
        q = self.fc2(q)

        return q

    def update(self, sarsa, q2, gamma):
     
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))
        
        q = self(s)
        '''In PyTorch we need to set the gradients to zero before starting to do backpropagation because PyTorch accumulates 
        the gradients on subsequent backward passes'''
        self.optimizer.zero_grad()
        '''Update rule for DDS'''
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[:,an] - q.cpu()[:,a],2)/2.0)
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
      
      
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()
  
def gym_act(env, q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return env.action_space.sample()
    else:
        avg = [i+j for i,j in zip(q1.cpu().data.numpy(),q2.cpu().data.numpy())]
        return np.argmax(np.array(avg))  

def gym_boxing():
    env = gym.make('Boxing-v0')

    obs_len = env.observation_space.shape[0]
    
    ddsarsa1 = DeepDoubleSarsa(obs_len, 18)
    ddsarsa1.to("cuda")
    ddsarsa2 = DeepDoubleSarsa(obs_len, 18)
    ddsarsa2.to("cuda")
    
    if LOAD:
      ddsarsa1.load("qa1.pt")
      ddsarsa2.load("qb1.pt")

    replay_buffer = ReplayBuffer(1000)
    alpha, gamma, epsilon = 0.1, 0.99, 1.0
    max_score = 0
    episodes = 5
    batch_size = 32
    rewards, lossa, lossb = [], [], []
    for e in range(1, episodes):
        done = False
        total_reward = 0

        obs = env.reset()
        obs = color.rgb2gray(obs)
        obs1 = cv2.resize(obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, 1, 84, 80)
        obs = obs.float()
        obs = obs.to("cuda")
        loss = Variable(torch.from_numpy(np.array([0.0])))
        qa = ddsarsa1(obs)
        qb = ddsarsa2(obs)
        a = gym_act(env, qa, qb, epsilon)
        while not done:

            n_obs, r, done, _ = env.step(a)
            total_reward +=r
            n_obs = color.rgb2gray(n_obs)
            n_obs1 = cv2.resize(n_obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
            n_obs = Variable(torch.from_numpy(n_obs1))
            n_obs = n_obs.view(-1, 1, 84, 80)
            n_obs = n_obs.float()
            n_obs = n_obs.to("cuda")
            n_qa = ddsarsa1(n_obs)
            n_qb = ddsarsa2(n_obs)
            an = gym_act(env, n_qa, n_qb, epsilon)
            if e<30000:
                epsilon -= 0.9/30000
            if done:
                replay_buffer.push(np.expand_dims(obs1, 0), a, r, np.expand_dims(n_obs1, 0), an, 1.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            else:
                replay_buffer.push(np.expand_dims(obs1, 0), a, r, np.expand_dims(n_obs1, 0), an, 0.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            
            if len(replay_buffer)>batch_size:
                s, a, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(batch_size)
                if np.random.rand(1)[0] > 0.5:
                    loss = ddsarsa1.update([s, a, r, sp, ap, d], q2nn, gamma)
                    lossa.append(loss.item())
                else:
                    loss = ddsarsa2.update([s, a, r, sp, ap, d], q1nn, gamma)
                    lossb.append(loss.item())
            
            obs1 = n_obs1
            a = an
        '''total reward for one episode'''
        if e==1:
            max_score = total_reward
        if SAVE and e%saving_fq==0:
            ddsarsa1.save("qa1.pt")
            ddsarsa2.save("qb1.pt")
            files.download("qa1.pt")
            files.download("qb1.pt")

        if SAVE and total_reward > max_score:
            if e>10000:
                ddsarsa1.save("max_qa1.pt")
                ddsarsa2.save("max_qb1.pt")
                files.download("max_qa1.pt")
                files.download("max_qb1.pt")
            max_score = total_reward


        print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.item()))
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

    plt.savefig('prova_boxing.png')
    plt.show()
    
gym_boxing()