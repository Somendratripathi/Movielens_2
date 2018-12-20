import numpy as np
import findspark
import collections
import pandas as pd
import os
import math


findspark.init()
from pyspark import SparkContext,SparkConf


def hash_function(x,a,b,n):
    return list(map(lambda x: (a*x+b)%n, x))


def signature_matrix(A, B, users):
    hash_users = collections.OrderedDict()
    for i, user in enumerate(users):
        idx = np.array(np.nonzero(np.array([row[i] for row in A]))[0])
        hash_users[user] = B[idx[:, None], :].min(0)[0]
    return np.array([hash_users[u] for u in users]).T


config = SparkConf().setAppName("LSH")

sc = SparkContext(conf=config)


train_path = 'Data/train_small.csv'


data = sc.textFile(train_path)
header = data.first()

'''
For us, the movies the user has seen is the items

So we need to identify which movies have been seen by the user

'''



# users = data.filter(lambda x: x!=header).map(lambda line: line.split(","))\
#     .map(lambda token: (int(token[0]),int(token[1]),float(token[2]))).groupBy(lambda line: line[0])\
#     .map(lambda x: list(x[1]))


data = data.filter(lambda x: x != header).map(lambda x: x.split(",")[:-1])\
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

user_mean_rating = data.map(lambda x: (x[0], x[2])).groupByKey()\
    .map(lambda x: (x[0],list(set(x[1])))).map(lambda x: (x[0], np.mean(x[1]))).collectAsMap()

user_movies = data.map(lambda x: (x[0], x[1])).groupByKey()\
    .map(lambda x: (x[0], list(set(x[1])))).sortByKey().collect()

print(user_movies)

# Identifying the unique movies and users
movies = data.map(lambda x: (int(x[1]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()
users = data.map(lambda x: (int(x[0]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()

print(movies)
print(len(movies))

# FOr each movie, seeing which user has seen the movie
movies_matrix = {}

for movie in movies:
    movies_matrix[movie] = [1 if movie in user[1] else 0 for user in user_movies]


# For each movie, computing the hash value for that
number_of_hash_functions = 60
number_of_bands = 20

hash_fns = []

for i in range(number_of_hash_functions):
    hash_fns.append(hash_function(movies, np.random.randint(0, 1000), np.random.randint(0, 1000), len(movies)))

hash_fns = np.array(hash_fns)
hash_fns = hash_fns.T

print(hash_fns.shape)

# Now for the user item matrix, creating the signature matrix

matrix = np.array([movies_matrix[movie] for movie in movies])

sig_matrix = signature_matrix(matrix, hash_fns, users)





