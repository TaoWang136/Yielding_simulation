import pickle
import random


def load_expert():
    with open('C:/Users/14487/python-book/驾驶员让行模拟论文/new_list_one.pkl', 'rb') as file:
        loaded_data_list = pickle.load(file)
    return loaded_data_list