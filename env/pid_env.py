import gym
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class PidEnvSingle(gym.Env):
    def __init__(self, sample_rate=1, setpoint=50):
        self.sample_rate = sample_rate
        self.setpoint = setpoint
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = 0
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        self.n = 250 # Simulation points
        self.done = 0
        self.xhistory = [0]
        self.yhistory = [0]
        self.history = []

    def step(self, action):
        self.kp = action[0] # Increasing p term reduces rise time
        self.ki = action[1]
        self.kd = action[2] # Increasing d term improves stability and decreases overshoot
        reward = 0
        last_rewards = deque(maxlen=5)
        completed = False
        curr_input = 0
        curr_speed = 0

        for i in range (250):
            # max x axis of n points 
            self.proportional = self.kp * self.error
            self.integral += self.error * self.sample_rate
            self.derivative = self.kd * (self.error - self.last_error) / self.sample_rate

            curr_input =  ((self.proportional + self.ki * self.integral + self.derivative))

            self.last_error = self.error
            self.currpoint += curr_input
            self.error = self.setpoint+30*i - self.currpoint
            last_rewards.append(reward)
            self.history.append(self.error)
            last_term = len(self.history)-1
            if last_term >= 1 and abs(self.history[last_term])<=0.001 and abs(self.history[last_term-1])<=0.001 and self.error < 0.001:
                completed = True
                break
            if abs(self.error) > 1000:
                break
        if completed == True:
            reward = 250-i-self.error
        else:
            reward = -abs(self.error) 


        return (self.error, self.setpoint), reward

    def reset(self, setpoint):
        self.history = []
        self.setpoint = setpoint
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = 0
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        return (self.error, self.setpoint)

    def render(self):
        print("*************************")
        print("Error: "+str(self.error))
        print("Proportional Term: "+str(self.proportional))
        print("Integral Term: "+str(self.integral))
        print("Derivative Term: "+str(self.derivative))
        print("Num steps required: "+str(len(self.history)))
        print("*************************")
        plt.plot(self.history)
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.show()
