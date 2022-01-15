import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(threshold=np.inf)

class Environment:

    # input example: Environment(<dataframe>)
    def __init__(self, df):

        # constant inputs
        self.durasilokasi = 2 * 3600  # 2 jam per hari
        self.durasihari = 13 * 3600  # 13 jam per hari
        self.durasimax = [1, 2, 3]  # 3 hari maks

        # custom inputs
        self.hotels = np.arange(28, 48)
        self.places = np.arange(0, 28)
        self.data = df #data lokasi (28x28) ditambah data hotelfrom dan hotel to ((20*28)*2)

        self.resetState()

        # RL model size
        self.actionsize = len(self.places) #semua tempat wisata (28 lokasi)
        self.statesize = self.data['from'].nunique()  # semua kemungkinan pergerakan (48 lokasi, hotel dan tempat wisata)

    def resetState(self, hotel=False, duration=False):
        # mutable inputs (state)
        if (not hotel):
            self.hotel = random.choice(self.hotels)
        else:
            self.hotel = hotel

        if (not duration):
            self.durasitour = random.choice(self.durasimax)  # max days for tour
        else:
            self.durasitour = duration

        self.state = self.hotel  # always start with the hotel
        self.places_not_done = self.places  # array that contain all posible next_place
        self.used_duration = 0  # initialize with 0, will be changed later
        self.duration_left = self.durasihari
        self.current_day = 1  # start with day 1
        self.reward = 0  # initialize default reward is 0
        self.next_state = self.hotel
        self.done = False
        return self.state

    def calcReward(self, optimum_result, worst_result):
        model_result = self.used_duration

        time_saved_by_action = worst_result - model_result
        time_waste_by_action = model_result - optimum_result

        reward = (time_saved_by_action - time_waste_by_action) / (worst_result - optimum_result) #dibagi agar proposional
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
        predicted_time_left = self.duration_left - predicted_duration - self.durasilokasi


        if (predicted_time_left > 0):
            if ((action in self.places_not_done) & (self.state != action)):
                self.reward += 5  # reward bertambah karena mengeksplorasi tempat baru
            else:
                self.reward -= 10  # reward berkurang karena kembali ke tempat yang sama
                action = random.choice(self.places_not_done)
                #print("!!RECURRING!!")

            self.next_state = action

            self.used_duration = self.calcDuration()
            self.duration_left = self.duration_left - self.used_duration - self.durasilokasi #durasi sisa dikurangi durasi tempat dan perjalanan

        elif ((len(self.places_not_done) == 0) | (self.current_day == self.durasitour)):
            self.done = True  # semua tempat dikunjungi / durasi tour habis

            # kembali ke hotel awal
            self.next_state = self.hotel
            self.reward += 10

            self.used_duration = self.calcDuration()
            self.duration_left = self.duration_left - self.used_duration - self.durasilokasi
            #print("**DAY IS TOUR**")
        else:
            self.current_day += 1

            #kembali ke hotel karena waktu dalam sehari tidak cukup
            self.next_state = self.hotel

            self.used_duration = self.calcDuration()
            self.duration_left = self.durasihari
            #print("BACK TO HOTEL")

        self.state = self.next_state
        self.places_not_done = self.places_not_done[self.places_not_done != self.next_state] #hilangkan tempat dari placesnotdone

        optimum, worst = self.findOptimumWorst()
        self.reward += self.calcReward(optimum, worst)
        return (self.next_state, self.reward, self.done)  # nextstate, state, reward, done

    def calcDuration(self):
        return self.data.loc[self.data['from'] == self.state].loc[self.data['to'] == self.next_state]['time'].values[0]

    def findOptimumWorst(self):
        filter = self.places_not_done

        temp = self.data.loc[self.data['from'] == self.state]
        temp['to'] = temp['to'].apply(lambda x: x if (x in filter) else None)
        temp = temp.dropna() #remove missing values
        # print(temp)

        optimum_result = temp['time'].min()
        worst_result = temp['time'].max()

        return optimum_result, worst_result

    def showCurrentCondition(self):
        print(f'Places Next: {self.state}')
        print(f'Places Left: {self.places_not_done}')
        print(f'Time left before next day: {self.duration_left}')
        print()
        print(f'Reward:  {round(self.reward,2)}')

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


class Agent():

  def __init__(self, env):
    self.action_size = env.actionsize
    print("Action size:", self.action_size) #28 tempat wisata

  def get_action(self, state):
    action = random.choice(range(self.action_size))
    return action


class QAgent(Agent):
  def __init__(self, env, discount_rate=0.97, learning_rate=0.001):
    super().__init__(env)
    self.state_size = env.statesize
    print("State size:", self.state_size) #48 kemungkinan

    self.eps = 1.0  # epsilon yang menentukan kerandoman aksi, semakin turun sehingga makin greedy
    self.min_eps = .1
    self.discount_rate = discount_rate #menentukan seberapa pengaruhnya prediksi state selanjutnya
    self.learning_rate = learning_rate #menentukan seberapa intens perubahan pada q table
    self.build_model()

  def build_model(self):
    self.q_table = np.loadtxt('qtable.csv', delimiter=',')
    #self.q_table = np.random.random([self.state_size, self.action_size])
    #self.q_table = np.round(self.q_table,2)

  def load_model(self, qtable):
    self.q_tale = qtable

  def get_action(self, state):
    q_state = self.q_table[state]
    action_greedy = np.argmax(q_state)
    action_random = super().get_action(state)
    return action_random if random.random() < self.eps else action_greedy

  def get_action_use(self, state):
    q_state = self.q_table[state]
    return np.argmax(q_state)

  def train(self, experience):
    state, action, next_state, reward, done = experience

    q_next = self.q_table[next_state] #atur nilai q_next menjadi nilai di q table pada next state (semua data to)
    q_next = np.zeros([self.action_size]) if done else q_next #atur nilai menjadi 0 jika done, mengurangi kemungkinan dia dipilih
    q_target = reward + self.discount_rate * np.max(q_next) #nilai reward + maksimal dari state tersebut * discount rate

    q_update = q_target - self.q_table[state, action] # jika target tidak bisa melebihi nilai yang ada di aksi state ke aksi, maka kurangi
    self.q_table[state, action] += self.learning_rate * q_update #update data pada q table state ke aksi dengan nilai q update

    if done:
      if (self.eps > self.min_eps):
        self.eps = self.eps * 0.99


env = Environment(df)
agent = QAgent(env)

rewards = []
eps = []
placedone= []
"""
total_avg_reward_per_day = 0
for ep in range(1000):
  total_reward = 0
  state = env.resetState()
  done = False
  #print("Durasi Tour: {}".format(env.durasitour))
  while not done:
    #print("*********")
    #print("Day: {}" .format(env.current_day))
    #print("*********")
    #print("State: {} ".format(state))

    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    agent.train((state,action,next_state,reward,done))
    state = next_state
    total_reward += reward

    #env.showCurrentCondition()
    placedone.append(state)
    #print("Total reward: {} ".format(round(total_reward,2)))
    #print()
    clear_output(wait=True)

  total_avg_reward_per_day += (total_reward/env.durasitour)
  eps.append(ep)
  rewards.append(total_avg_reward_per_day)

  #print("--------------------------------------------------------------------------------------")
  #print("--------------------------------------------------------------------------------------")

  #print("Done: {},".format(placedone))
  #time.sleep(0.01)
  placedone.clear()
  total_avg_reward_per_day = 0
  clear_output(wait=True)

  #print("--------------------------------------------------------------------------------------")
  #print("--------------------------------------------------------------------------------------")

print("Episode: {}, Total average reward: {}, eps: {}".format(ep,total_avg_reward_per_day,agent.eps))
print(agent.q_table)

#save q table
np.savetxt('qtable.csv', agent.q_table, delimiter=',')
"""

#build rute
test_env = Environment(df)
test_agent = QAgent(test_env)
qtable = np.loadtxt('qtable.csv', delimiter=',')
test_agent.load_model(qtable)

hotel = int (input("Hotel Number: "))
duration = int (input("Tour Duration: "))
#hotel = 30
#duration = 3
action_list = []

total_reward = 0
state = test_env.resetState(hotel, duration)
done = False

action_list.append(state)
while not done:

  print("Day: {}".format(test_env.current_day))
  action = test_agent.get_action_use(state)
  next_state, reward, done = test_env.step(action)
  state = next_state

  action_list.append(state)
  print(f'Places Next: {test_env.state}')
  print(f'Reward: {test_env.reward}')
  #test_env.showCurrentCondition()
  total_reward += test_env.reward

print("*********")
print(f'Action List: {action_list}')
print(f'Total reward: {total_reward}')