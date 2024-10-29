import matplotlib
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time


class FrozenLakeWrapper:
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")
        q_table = np.zeros((self.env.observation_space.n,self.env.action_space.n))
        self.q_table = q_table
        self.rewards = []
        print("Qtable:\n",self.q_table)

    def train(self, num_episodes=5000, learning_rate=0.2,max_steps=100,gamma=0.99,epsilon=1,epsilon_decay=0.9995, verbose=0):
        episode_steps=[]
        for episode in range(num_episodes):

            state = self.env.reset()[0]
            step = 0
            done = False
            total_rewards = 0
            while not done:
                step=step+1
                # print("episode: ", episode," step: ", step)
                if random.uniform(0,1) > epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, truncated, info = self.env.step(action)

                max_new_state = np.max(self.q_table[new_state, :])

                self.q_table[state, action] = self.q_table[state,action] + learning_rate * (reward + gamma * max_new_state - self.q_table[state,action])

                total_rewards += reward
                state = new_state

                if verbose >= 3:
                    print("episode:{}, steps:{}, current_reward:{}".format(episode,step,total_rewards))

                if done:
                    continue

            epsilon = epsilon_decay * epsilon
            self.rewards.append(total_rewards)
            if verbose >= 2:
                print("Episode:{},steps:{},reward:{}".format(episode, step, total_rewards))
        if verbose >= 1:
            print("Score:{} , num episodes: {}, avg steps: {}, minSuccessfulSteps: {} ", str(sum(self.rewards)/num_episodes))
            print(self.q_table)

    def one_game(self):
        state = self.env.reset()[0]
        done = False
        total_rewards = 0
        step = 0

        while not done:
            img = self.env.render()
            plt.imshow(img)
            plt.axis('off')  # Hide axes for a cleaner view
            action = np.argmax(self.q_table[state, :])
            new_state, reward, done, truncated, info = self.env.step(action)
            total_rewards += reward
            state = new_state
            plt.pause(1)
            plt.clf()

        print("Num steps:{}, Reward:{}".format(step, total_rewards))


fw = FrozenLakeWrapper()
fw.train(verbose=2)
fw.one_game()