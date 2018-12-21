import numpy as np
import findspark
import collections
from collections import ChainMap
from functools import reduce
import pandas as pd
import time
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


def predict(user, item , user_mean_rating, output_scores, user_movies):
    if user not in user_mean_rating:
        return (3,1)
    if user not in output_scores:
        return (user_mean_rating[user],1)
    num = 0
    den = 0
    similar_users = list(output_scores[user].keys())
    for similar_user in similar_users:
        if item in user_movies[similar_user].keys():
            num += output_scores[user][similar_user] * user_movies[similar_user][item]
            den += np.abs(output_scores[user][similar_user])
    if den == 0:
        return (user_mean_rating[user],1)
    return (num/den,0)


def rmse(x,y):
    z = np.square([a-b for a,b in zip(x,y)])
    return math.sqrt(np.sum(z)/len(z))


def precision(x,y):
    cnt = np.sum([a & b for a,b in zip(x,y)]) * 1.0
    return cnt/np.sum(y)


config = SparkConf().setAppName("LSH")

sc = SparkContext(conf=config)


train_path = '../Data/train.csv'


data = sc.textFile(train_path)
header = data.first()

'''
For us, the movies the user has seen is the items

So we need to identify which movies have been seen by the user

'''


data = data.filter(lambda x: x != header).map(lambda x: x.split(",")[:-1])\
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

user_mean_rating = data.map(lambda x: (x[0], x[2])).groupByKey()\
    .map(lambda x: (x[0],list(set(x[1])))).map(lambda x: (x[0], np.mean(x[1]))).collectAsMap()

user_movies = data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()\
    .map(lambda x: (x[0], dict(list(set(x[1]))))).sortByKey().collectAsMap()


# Identifying the unique movies and users
movies = data.map(lambda x: (int(x[1]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()
users = data.map(lambda x: (int(x[0]), 1)).groupByKey().sortByKey().map(lambda x: x[0]).collect()

print(movies)
print(len(movies))

# FOr each movie, seeing which user has seen the movie
movies_dictionary = collections.OrderedDict()

for movie in movies:
    movies_dictionary[movie] = [1 if movie in user_movies[user].keys() else 0 for user in user_movies]

matrix = np.array([movies_dictionary[movie] for movie in movies])

# For each movie, computing the hash value for that
number_of_hash_functions = 126
number_of_bands = 42
threshold = 0.05

hash_fns = []

t1 = time.time()
for i in range(number_of_hash_functions):
    np.random.seed(i)
    a = np.random.randint(0, 1000)
    b = np.random.randint(0, 1000)
    hash_fns.append(hash_function(movies, a, b, len(movies)))

hash_fns = np.array(hash_fns)
hash_fns = hash_fns.T

print(hash_fns.shape)


# Now for the user item matrix, creating the signature matrix


sig_matrix = signature_matrix(matrix, hash_fns, users)

print(sig_matrix.shape)

# print(sig_matrix)


sig_matrix_list = sig_matrix.tolist()

print(len(sig_matrix_list[0]), len(sig_matrix_list))


sig_matrix = sc.parallelize(sig_matrix_list, number_of_bands)

find_users = sig_matrix.mapPartitions(lambda x: get_similar_users(x, users))\
    .groupByKey().map(lambda x: (x[0], (list(set([item for sublist in x[1] for item in sublist]))))).sortByKey()

print(find_users.take(20))
print(len(find_users.collect()))

similarity_scores = find_users.map(lambda x: jaccard_similarity(x[0], x[1], user_movies))\
    .flatMap(lambda x: x)

t2 = time.time()

print(similarity_scores.take(20))

similarity_scores = similarity_scores.filter(lambda x: x[1] > threshold)\
    .map(lambda x: (x[0][0], [(x[0][1],x[1])])).groupByKey().map(lambda x: (x[0], list(x[1])))\
    .map(lambda x: (x[0], dict([y[0] for y in x[1]]))).sortByKey()

print(similarity_scores.take(20))

output_scores = similarity_scores.collectAsMap()

print(output_scores[1])

test_path = '../Data/test.csv'
test_data = sc.textFile(test_path)
header = test_data.first()

test_data = test_data.filter(lambda x: x != header).map(lambda x: x.split(",")[:-1])\
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

count = len(test_data.collect())

test_data_prediction = test_data.map(lambda x: predict(x[0], x[1],
                                                       user_mean_rating, output_scores, user_movies))


test_data_actual = test_data.map(lambda x: x[2]).collect()

issues_with_data = np.sum(test_data_prediction.map(lambda x: x[1]).collect())

test_data_prediction = test_data_prediction.map(lambda x: x[0]).collect()

print(issues_with_data/count)
# print(test_data_prediction)
# print(test_data_actual)


print(rmse(test_data_prediction, test_data_actual))

movie_to_recommend_level = 3.5

# I will define Precision as the number of movies which were above a threshold in the testdata
# and how many of them were we able to accurately capture

# I will test the model for all values above the threshold rather than top K since choosing K items might be tough if
# we have more no of choices. We don't want to penalize the model if it correctly captured both. Choosing the K
# can be done randomly


test_data_predicted_recommending = [x >= movie_to_recommend_level for x in test_data_prediction]
test_data_actual_recommending = [x >= movie_to_recommend_level for x in test_data_actual]


print(precision(test_data_actual_recommending, test_data_predicted_recommending))






