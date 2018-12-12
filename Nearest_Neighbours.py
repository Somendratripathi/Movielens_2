import numpy as np
import findspark
import pandas as pd


findspark.init()
from pyspark import SparkContext,SparkConf


def hash_function(x, a, b, m):
    value = (a*x + b)%m
    return value


train_path = 'Data/train.csv'
data = pd.read_csv(train_path)

unique_movies = sorted(list(set(data['MovieID'])))
unique_users = sorted(list(set(data['UserID'])))


user_movies = {}
movies_users = {}

for user in unique_users:
    movies_for_user = list(data[data.UserID == user]['MovieID'])
    user_movies[user] = movies_for_user

for movie in movies_users:
    users_seeing_movie = list(data[data.MovieID == movie]['UserID'])
    movies_users[movie] = users_seeing_movie


number_of_hash_functions = 60
number_of_bands = 20

'''
Now we need to generate a,b that would out hash functions. We are defining 60 hash functions
'''

np.random.seed(3)
a_list = np.random.randint(1000, size=60)
np.random.seed(5)
b_list = np.random.randint(1000, size=60)

hash_values = []

n = len(unique_movies)

for (a, b) in zip(a_list, b_list):
    value = {}
    for movie in unique_movies:
        value[movie] = hash_function(movie, a, b, n)
    hash_values += [value]


matrix = np.zeros((number_of_hash_functions, len(unique_users)))

for index, hash_value in enumerate(hash_values):
    for user in unique_users:
        min_value = np.min([hash_value[i] for i in user_movies[user]])
        matrix[index][user-1] = min_value





# config = SparkConf().setAppName("LSH")
#
# sc = SparkContext(conf=config)
#
#
# train_path = 'Data/train.csv'
#
#
# data = sc.textFile(train_path)
# header = data.first()

'''
For us, the movies the user has seen is the items

So we need to identify which movies have been seen by the user

'''


# users = data.filter(lambda x: x!=header).map(lambda line: line.split(","))\
#     .map(lambda token: (int(token[0]),int(token[1]),float(token[2]))).groupBy(lambda line: line[0])\
#     .map(lambda x: list(x[1]))
#
#
#
# print(users.first())
