import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
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
        self.actionsize = len(self.places)
        self.statesize = self.data['from'].nunique()  # all possible current locations

    def resetState(self):
        # mutable inputs (state)
        self.hotel = random.choice(self.hotels)
        self.tour_duration = random.choice(self.possible_duration)  # max days for tour
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

        # if (action < 28):
        predicted_duration = \
        self.data.loc[self.data['from'] == self.state].loc[self.data['to'] == action]['time'].values[0]
        # else:
        # go to hotel from hotel, terminate with fatal negative rewards
        # return (action, -50, True) #nextstate, state, reward, done

        predicted_time_left = self.duration_left - predicted_duration - self.duration_each_location

        if (predicted_time_left > 0):
            if ((action in self.places_not_done) & (self.state != action)):
                self.reward += 5  # because it can reach new place
            else:
                self.reward -= 50  # because not move from current place or visit same place or go to hotel
                self.done = True

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

