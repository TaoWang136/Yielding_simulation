import pickle
import random
import pandas as pd


def load_expert():
    expert = pd.read_csv(
        'C:/Users/14487/python-book/驾驶员让行模拟论文/第一轮修改/expert.csv',
        index_col=0
    )
    print('使用')
    return expert