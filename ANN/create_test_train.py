import pandas as pd
import random

'''
Splitting the data in a 80:20 ratio
'''

ratio = 0.8
path = '../Data/ratings.csv'
df = pd.read_csv(path)

numbers = [x for x in range(0, df.shape[0])]
random.seed(3)
randomList = random.sample(numbers, int(ratio * len(numbers)))

train = df.iloc[randomList]

other_indices = df.index.isin(randomList)
test = df[~other_indices]

train.to_csv('../Data/train_1.csv', index=False)
test.to_csv('../Data/test_1.csv', index=False)


