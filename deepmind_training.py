import numpy as np 
import random
import operator as op
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym
from skimage import color
import cv2


LOAD = False
SAVE = True
saving_fq = 100
epsilon_start, epsilon_stop, epsilon_decay = 1.0, 0.1, 400
episodes = 500
batch_size = 128
BUFFER_SIZE = 1000
target_update = 50
squeeze = 4
cuda0 = torch.device('cuda')

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
        # self.cn1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=1)
        # self.cn1b = torch.nn.BatchNorm2d(16)
        # self.cn1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=1)
        # self.cn1b = torch.nn.BatchNorm2d(16)
        # self.cn2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        # self.cn2b = torch.nn.BatchNorm2d(32)
        # self.cn3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.cn3b = torch.nn.BatchNorm2d(64)
        
        self.fc1 = torch.nn.Linear(initus, 32, bias=bias)
        # self.fc2 = torch.nn.Linear(64, 64, bias=bias)

        # self.fc3 = torch.nn.Linear(64, 64, bias=bias)
        # self.fc1b = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(32, exitus, bias=bias)
        
        

        ''' torch.optim is a package implementing various optimization algorithms '''
        # Adam optimizer is one of the most popular gradient descent optimizer in deep learning
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)

    def forward(self, input):
        # q = torch.nn.functional.relu(self.cn1b(self.cn1(input)))
        # q = torch.nn.functional.relu(self.cn2b(self.cn2(q)))
        # q = torch.nn.functional.relu(self.cn3b(self.cn3(q)))
        # q = q.view(-1, self.num_flat_features(q))

        ''' Activation function (to get the output of node) : ReLU function returns a 0 for input values less than 0,
         while input values above 0, the function returns a value between 0 and 1 '''
        # q = input
        # q = q.view(-1, self.num_flat_features(q))
        q = torch.nn.functional.relu(self.fc1(input))
        # q = torch.nn.functional.relu(self.fc2(q))
        # q = torch.nn.functional.relu(self.fc1(q))
        # q = torch.nn.functional.softmax(self.fc2(q), dim=1)
        q = self.fc2(q)

        return q

    def update(self, sarsa, q2, gamma):
     
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))
        
        # q = self(s.view(-1, 1, 84, 80))
        q = self(s)

        in_q = [np.arange(len(a)), a]
        in_qb = [np.arange(len(an)), an]
        self.optimizer.zero_grad()
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[in_qb] - q.cpu()[in_q],2)/2.0)
        loss.backward()
        self.optimizer.step()
        return loss
      
    def num_flat_features(self, x):
        size = x.size()[1:]
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
        avg = [i+j for i,j in zip(np.squeeze(q1.cpu().data.numpy()),np.squeeze(q2.cpu().data.numpy()))]
        return np.argmax(np.array(avg))  

def deepmind_training():
    env = gym.make('CartPole-v1')
    obs_len = env.observation_space.shape[0]
    ddsarsa = DeepDoubleSarsa(obs_len, 2, bias=True)
    ddsarsa.to(device=cuda0)
    target = DeepDoubleSarsa(obs_len, 2, bias=True)
    target.to(device=cuda0)
    if LOAD:
      ddsarsa.load("bowling_fc0a.pt")
      target.load("bowling_fc0b.pt")

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    alpha, gamma, epsilon = 0.1, 0.99, 1.0
    max_score = 0
    rewards, lossa, lossb = [], [], []

    for e in range(1, episodes):
        done = False
        total_reward = 0

        obs1 = env.reset()
        # obs = color.rgb2gray(obs)
        # obs1 = cv2.resize(obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        obs = obs.to(device=cuda0)
        qa = ddsarsa(obs)
        qb = target(obs)
        a = gym_act(env, qa, qb, epsilon)

        loss = Variable(torch.from_numpy(np.array([0.0])))
        
        t = 0
        while not done:
            
            n_obs1, r, done, _ = env.step(a)
            total_reward +=r
            # n_obs = color.rgb2gray(n_obs)
            # n_obs1 = cv2.resize(n_obs, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
            n_obs = Variable(torch.from_numpy(n_obs1))
            n_obs = n_obs.view(-1, obs_len)
            n_obs = n_obs.float()
            n_obs = n_obs.to(device=cuda0)
            n_qa = ddsarsa(n_obs)
            n_qb = target(n_obs)
            an = gym_act(env, n_qa, n_qb, epsilon) 
            # if not t%1:           
            if done:
                replay_buffer.push(obs1, a, r, n_obs1, an, 1.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            else:
                replay_buffer.push(obs1, a, r, n_obs1, an, 0.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            a = an
            if len(replay_buffer)>=batch_size:
                # if np.random.randn(1)[0] > 0.5:
                if e%target_update:
                    s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(batch_size)
                    loss = ddsarsa.update([s, ac, r, sp, ap, d], q2nn, gamma)
                    lossa.append(loss.item())
                else:
                    # s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(batch_size)
                    # loss = target.update([s, ac, r, sp, ap, d], q1nn, gamma)
                    # lossb.append(loss.item())
                    ddsarsa.save('target.pt')
                    target.load('target.pt')

            obs1 = n_obs1
            t += 1

        if e<epsilon_decay:
                epsilon -= 1.0/epsilon_decay
        rewards.append(total_reward)

        if e==1:
            max_score = total_reward
        if SAVE and e%saving_fq==0:
            ddsarsa.save("models/cartpole_dm1a.pt")
            target.save("models/cartpole_dm1b.pt")
            np.save('models/cartpole_fc1_rewards', rewards)
            np.save('models/cartpole_fc1_lossa', lossa)

        if SAVE and total_reward > max_score:
            if e>epsilon_decay:
                ddsarsa.save("models/max_dma.pt")
                target.save("models/max_dmb.pt")
            max_score = total_reward
        
        print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.item()))
        rewards.append(total_reward)

    plt.subplot(2,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(2,1,2)
    plt.plot(lossa)
    plt.title("Lossa")
    plt.savefig('prova_cartpole.png')
    plt.show()

def deepmind_bowling():
    env = gym.make('Bowling-v0')
    # obs_len = env.observation_space.shape[0]
    ddsarsa = DeepDoubleSarsa(None, 6, bias=True)
    ddsarsa.to(device=cuda0)
    target = DeepDoubleSarsa(None, 6, bias=True)
    target.to(device=cuda0)
    if LOAD:
      ddsarsa.load("bowling_fc0a.pt")
      target.load("bowling_fc0b.pt")

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    gamma, epsilon = 0.99, epsilon_start
    max_score = 0
    rewards, lossa, lossb = [], [], []

    for e in range(1, episodes):
        done = False
        total_reward = 0

        obs1 = env.reset()
        obs1 = color.rgb2gray(obs1)
        obs1 = cv2.resize(obs1, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, 1, 84, 80)
        obs = obs.float()
        obs = obs.to(device=cuda0)
        qa = ddsarsa(obs)
        qb = target(obs)
        a = gym_act(env, qa, qb, epsilon)

        loss = Variable(torch.from_numpy(np.array([0.0])))
        sq_r = 0
        t = 0
        ep_lossa = 0
        ep_lossb = 0

        while not done:
            # if e%100==0:
            #     env.render()
            n_obs1, r, done, _ = env.step(a)
            total_reward +=r
            n_obs1 = color.rgb2gray(n_obs1)
            n_obs1 = cv2.resize(n_obs1, None, fx = 0.5, fy = 0.4, interpolation =cv2.INTER_CUBIC)
            n_obs = Variable(torch.from_numpy(n_obs1))
            n_obs = n_obs.view(-1, 1, 84, 80)
            n_obs = n_obs.float()
            n_obs = n_obs.to(device=cuda0)
            n_qa = ddsarsa(n_obs)
            n_qb = target(n_obs)
            an = gym_act(env, n_qa, n_qb, epsilon) 
            sq_r += r
            if not t%squeeze or done:           
                if done:
                    replay_buffer.push(obs1, a, sq_r, n_obs1, an, 1.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
                else:
                    replay_buffer.push(obs1, a, sq_r, n_obs1, an, 0.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
                sq_r = 0
                obs1 = n_obs1
                a = an
            if len(replay_buffer)>batch_size:
                # if np.random.randn(1)[0] > 0.5:
                if e%target_update:
                    s, ac, r, sp, ap, d, _, q2nn = replay_buffer.sample(batch_size)
                    loss = ddsarsa.update([s, ac, r, sp, ap, d], q2nn, gamma)
                    ep_lossa += loss.item()
                else:
                    # s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(batch_size)
                    # loss = target.update([s, ac, r, sp, ap, d], q1nn, gamma)
                    # ep_lossb += loss.item()
                    ddsarsa.save('target.pt')
                    target.load('target.pt')

            t += 1
        lossa.append(ep_lossa/t)
        lossb.append(ep_lossb/t)
        rewards.append(total_reward)

        if e<epsilon_decay:
                epsilon -= (epsilon_start - epsilon_stop)/epsilon_decay

        if e==1:
            max_score = total_reward
        if SAVE and e%saving_fq==0:
            ddsarsa.save("models/bowling_dm4a.pt")
            target.save("models/bowling_dm4b.pt")
            np.save('models/bowling_cn3_rewards', rewards)
            np.save('models/bowling_cn3_lossa', lossa)
            np.save('models/bowling_cn3_lossb', lossb)


        if SAVE and total_reward > max_score:
            if e>epsilon_decay:
                ddsarsa.save("models/bowling_max_dm4a.pt")
                target.save("models/bowling_max_dm4b.pt")
            max_score = total_reward
        
        print("Episode: {} | Timesteps: {} | Reward: {} | Loss: {}".format(e, t, total_reward, np.mean(ep_lossa/t + ep_lossb/t)))

    plt.subplot(2,1,1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(2,1,2)
    plt.plot(lossa)
    plt.title("Lossa")
    plt.savefig('prova_boxing.png')
    plt.show()

if __name__ == "__main__":
    deepmind_training()
    # deepmind_bowling()