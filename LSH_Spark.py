import numpy as np
import findspark
import collections
from functools import reduce
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


def get_similar_users(rows, user):
    similar_users = []
    transpose_of_rows = np.array([row for row in rows]).T
    # print(transpose_of_rows.T.shape)
    for i, row in enumerate(user):
        val = transpose_of_rows[i, :]
        x = np.where((transpose_of_rows == tuple(val)).all(axis=1))[0].tolist()
        similar_users = similar_users + [(user[i], [user[j] for j in x])]
    return iter(similar_users)


def jaccard_similarity(target_user, similar_users, user_movies):
    values = []
    compare_by = set(user_movies[target_user])
    for user in similar_users:
        if user != target_user:
            score = len(set(user_movies[user]).intersection(compare_by)) * 1.0
            score = score / len(set(user_movies[user]).union(compare_by))
            values += [((target_user, user), score)]
    return values


# l = [[1], [2]]
# z = set([item for sublist in l for item in sublist])
# print(z)
# exit(0)

# A = np.array([[1,2,1], [3,4,3]])
#
# for row in A.T:
#     val = row
#     x = np.where((A.T == tuple(val)).all(axis=1))[0].tolist()
#     print(x)
#
# exit(0)
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
    .map(lambda x: (x[0], list(set(x[1])))).sortByKey().collectAsMap()


# Identifying the unique movies and users
movies = data.map(lambda x: (int(x[1]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()
users = data.map(lambda x: (int(x[0]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()

print(movies)
print(len(movies))

# FOr each movie, seeing which user has seen the movie
movies_matrix = {}

for movie in movies:
    movies_matrix[movie] = [1 if movie in user_movies[user] else 0 for user in user_movies]


# For each movie, computing the hash value for that
number_of_hash_functions = 60
number_of_bands = 20
threshold = 0.1

hash_fns = []

for i in range(number_of_hash_functions):
    np.random.seed(i)
    a = np.random.randint(0, 1000)
    b = np.random.randint(0, 1000)
    hash_fns.append(hash_function(movies, a, b, len(movies)))

hash_fns = np.array(hash_fns)
hash_fns = hash_fns.T

print(hash_fns.shape)


# Now for the user item matrix, creating the signature matrix

matrix = np.array([movies_matrix[movie] for movie in movies])

sig_matrix = signature_matrix(matrix, hash_fns, users)

print(sig_matrix.shape)

print(sig_matrix)

sig_matrix_list = sig_matrix.tolist()

print(len(sig_matrix_list[0]), len(sig_matrix_list))


sig_matrix = sc.parallelize(sig_matrix_list, number_of_bands)

find_users = sig_matrix.mapPartitions(lambda x: get_similar_users(x, users))\
    .groupByKey().map(lambda x: (x[0], (list(set([item for sublist in x[1] for item in sublist]))))).sortByKey()

print(find_users.take(20))

print(len(find_users.collect()))

similarity_scores = find_users.map(lambda x: jaccard_similarity(x[0], x[1], user_movies))\
    .flatMap(lambda x: x)

# print(similarity_scores.take(20))

similarity_scores = similarity_scores.filter(lambda x: x[1] > threshold)

print(similarity_scores.take(20))



