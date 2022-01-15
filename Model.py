import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import datetime
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")


class Environment:

    # input example: Environment(<dataframe>)
    def __init__(self, df):

        # constant inputs
        self.duration_each_location = 2 * 3600  # 2 hours for each place
        self.duration_per_day = 13 * 3600  # 13 hours for each day
        self.possible_duration = [1, 2, 3]  # 1day, 2days, 3days

        # custom inputs
        self.hotels = np.arange(28, 48)
        self.places = np.arange(0, 28)
        self.data = df

        self.resetState()

        # RL model size
        self.actionsize = len(self.places) + 1
        self.statesize = self.data['from'].nunique()  # all possible current locations

    def resetState(self, hotel=False, duration=False):
        # mutable inputs (state)
        if (not hotel):
            self.hotel = random.choice(self.hotels)
        else:
            self.hotel = hotel

        if (not duration):
            self.tour_duration = random.choice(self.possible_duration)  # max days for tour
        else:
            self.tour_duration = duration

        self.state = self.hotel  # always start with the hotel
        self.places_not_done = self.places  # array that contain all posible next_place
        self.used_duration = 0  # initialize with 0, will be changed later
        self.duration_left = self.duration_per_day
        self.current_day = 1  # start with day 1
        self.reward = 0  # initialize default reward is 0
        self.next_state = self.hotel
        self.done = False
        return self.state

    def calcReward(self, optimum_result, worst_result):
        model_result = self.used_duration

        time_saved_by_action = worst_result - model_result
        time_waste_by_action = model_result - optimum_result

        # print(f"\t -optimum: {optimum_result}")
        # print(f"\t -worst: {worst_result}")
        # print(f"\t -model: {model_result}")

        reward = (time_saved_by_action - time_waste_by_action) / (worst_result - optimum_result)
        return reward

    def step(self, action):
        if (action == 28):
            action = self.hotel
            if (action == self.state):
                self.done = True
                self.reward -= 50
                return (action, self.reward, self.done)  # nextstate, state, reward, done

        predicted_duration = \
        self.data.loc[self.data['from'] == self.state].loc[self.data['to'] == action]['time'].values[0]
        predicted_time_left = self.duration_left - predicted_duration - self.duration_each_location

        if (predicted_time_left > 0):
            if ((action in self.places_not_done) & (self.state != action)):
                self.reward += 5  # because it can reach new place
            else:
                self.reward -= 10  # because not move from current place or visit same place or go to hotel
                action = random.choice(self.places_not_done)

            self.next_state = action

            self.used_duration = self.calcDuration()
            self.duration_left = self.duration_left - self.used_duration - self.duration_each_location

        elif ((len(self.places_not_done) == 0) | (self.current_day == self.tour_duration)):
            self.done = True  # all places visited or time up

            # go back to hotel for the last day
            self.next_state = self.hotel

            self.used_duration = self.calcDuration()
            self.duration_left = self.duration_left - self.used_duration - self.duration_each_location

        else:
            self.current_day += 1

            # go back to hotel time since it's action aborted as the duration left is not enough
            self.next_state = self.hotel

            self.used_duration = self.calcDuration()
            self.duration_left = self.duration_per_day

        # print(f"from {self.state} to {self.next_state}")

        self.state = self.next_state
        self.places_not_done = self.places_not_done[self.places_not_done != self.next_state]

        optimum, worst = self.findOptimumWorst()
        self.reward += self.calcReward(optimum, worst)
        return (self.next_state, self.reward, self.done)  # nextstate, state, reward, done

    def calcDuration(self):
        return self.data.loc[self.data['from'] == self.state].loc[self.data['to'] == self.next_state]['time'].values[0]

    def findOptimumWorst(self):
        filter = self.places_not_done

        temp = self.data.loc[self.data['from'] == self.state]
        temp['to'] = temp['to'].apply(lambda x: x if (x in filter) else None)
        temp = temp.dropna()
        # print(temp)

        optimum_result = temp['time'].min()
        worst_result = temp['time'].max()

        return optimum_result, worst_result

    def showCurrentCondition(self):
        print(f'Places Now: {self.state}')
        print(f'Places Left: {self.places_not_done}')
        print(f'Time left before next day: {self.duration_left}')
        print(f'Day: {self.current_day}')


#  def printData():
#    print(f"Hotels: { hotels } \n\n")
#    print(f"Tour objects: { places } \n\n")
#    print(f"Tour duration: { tour_duration }")
#    print(f"Known data: \n { data.head() } \n\n")
#    print(f"Hotel - Location Data: \n { hotel_loc_data.head() } \n\n")

# real data

#from hotel to places
hotel_loc_data = pd.read_csv('hotel-loc.csv', index_col=[0], names=['No', 'from', 'to', 'time'])
hotel_loc_data2 = pd.read_csv('hotel-loc.csv', index_col=[0], names=['No', 'from', 'to', 'time'])

# 28 - 47 is for HOTEL
hotel_loc_data['from'] = hotel_loc_data['from'] + 27
hotel_loc_data2['to'] = hotel_loc_data['from']
hotel_loc_data2['from'] = hotel_loc_data['to']

#from places to places
data = pd.read_csv('loc-loc.csv', index_col=[0], names=['No', 'from', 'to', 'time'])
#data = data[data['from'] != data['to']]

df = data.append(hotel_loc_data, ignore_index=True)
df = df.append(hotel_loc_data2, ignore_index=True)


hotelList = pd.read_csv('datahotel.csv', names=['code', 'name', 'lat', 'long', 'address'])
hotelList = hotelList.loc[:, ['code', 'name']]

hotelList['code'] = hotelList['code'] + 27


locList = pd.read_csv('datalokasi.csv', names=['code', 'name', 'lat', 'long', 'a', 'address', 'b', '', 'c', 'd', 'e'])
locList = locList.loc[:, ['code', 'name']]

df_all_loc = locList.append(hotelList, ignore_index=True)
df_all_loc


class Agent():

    def __init__(self, env):
        self.action_size = env.actionsize
        print("Action size:", self.action_size)

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.001):
        super().__init__(env)
        self.state_size = env.statesize
        print("State size:", self.state_size)

        self.eps = 1.0
        self.min_eps = .1
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])

    def load_model(self, qtable):
        self.q_tale = qtable

    def get_action_use(self, state):
        q_state = self.q_table[state]
        return np.argmax(q_state)

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            if (self.eps > self.min_eps):
                self.eps = self.eps * 0.999


env = Environment(df)
agent = QAgent(env)

rewards = []
eps = []

total_avg_reward_per_day = 0
for ep in range(0,3000):
    total_reward = 0
    state = env.resetState()
    done = False
    while not done:
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    agent.train((state,action,next_state,reward,done))
    state = next_state
    total_reward += reward

    #print(agent.q_table)
    clear_output(wait=True)

    total_avg_reward_per_day = (total_reward/env.tour_duration)
    eps.append(ep)
    rewards.append(total_avg_reward_per_day)

    print("Episode: {}, Total reward: {}, eps: {}".format(ep,total_avg_reward_per_day,agent.eps))
    #time.sleep(0.05)
    clear_output(wait=True)


    np.savetxt('qtable.csv', agent.q_table, delimiter=',')


#Print penjadwalan rute
n = 0

hotel_name = action_name = df_all_loc.loc[action_list[0]]['name']

for a in range(len(action_list[:-1])):
  action_now = action_list[a]
  action_name = df_all_loc.loc[action_now]['name']
  action_next = action_list[a+1]

  if (action_now>27):
    time_now = datetime.datetime(year=2020, month=12, day=9, hour=7, minute=0, second=0) #7.00 - 19.00 (estimate)
    n += 1
    print(f'\n[Hotel - day {n}]')


  time_used = df.loc[df['from'] == action_now].loc[df['to'] == action_next]['time'].values[0]
  time_now += datetime.timedelta(0, int (time_used)) # days, seconds, then other fields.

  if (action_now != action_list[0]): #if not hotel, then have visit duration
    time_now += datetime.timedelta(0,7200) #2 hours

  if (action_list[a+1] > 27):
    print(action_name, end=f'\n ({time_now.time()}) BACK TO HOTEL\n')
  else:
    print(f'{action_name}', end=f' \n ({time_now.time()}) ')