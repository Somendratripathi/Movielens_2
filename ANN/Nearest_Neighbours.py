import numpy as np
import findspark
import pandas as pd
import os
import math


findspark.init()
from pyspark import SparkContext,SparkConf


def hash_function(x, a, b, m):
    value = (a*x + b)%m
    return value


def neighbours_of_user(user_index, hash_matrix, num_of_bands, threshold, user_dict):
    user_columns = np.split(hash_matrix[:, user_index - 1], num_of_bands)
    matrix_split = hash_matrix.copy()
    matrix_split = np.delete(matrix_split, [user_index - 1], axis=1)
    similar_users = []
    for index, column in enumerate(matrix_split.T):
        if index < user_index - 1:
            value = index + 1
        else:
            value = index + 2
        if value in user_dict:
            if user_index in user_dict[value]:
                similar_users += [value]
        else:
            split_of_column = np.split(column, num_of_bands)
            count = np.sum([x == y for x,y in zip(split_of_column, user_columns)]) * 1.0 /num_of_bands
            if count >= threshold:
                similar_users += [value]
    return similar_users


def cosine_similarity(user_1, user_2, user_movies):
    common_movies = set(user_movies[user_1].keys()).intersection(set(user_movies[user_2].keys()))
    ratings_user_1 = [user_movies[user_1][key] for key in common_movies]
    ratings_user_2 = [user_movies[user_2][key] for key in common_movies]
    num = np.dot(ratings_user_1, ratings_user_2)
    den = math.sqrt(np.sum(np.square(ratings_user_1))) * math.sqrt(np.sum(np.square(ratings_user_2)))
    return num/den


def predict_rating(user, movie, cosine_numbers, movies_users, user_neighbours, user_movies):
    try:
        similar_users_who_watched_movie = sorted(set(movies_users[movie]).intersection(set(user_neighbours[user])))
    except Exception:
        print("error", user, movie)
        return np.average([user_movies[user][movie] for movie in user_movies[user]])
    if len(similar_users_who_watched_movie) == 0:
        print("No user found", user, movie)
        return np.average([user_movies[user][movie] for movie in user_movies[user]])
    movie_rating = np.array([user_movies[similar_user][movie] for similar_user in similar_users_who_watched_movie])
    cosine_rating = np.array([cosine_numbers[user][similar_user] for similar_user in similar_users_who_watched_movie])
    rating = np.dot(movie_rating, cosine_rating)/len(similar_users_who_watched_movie)
    if np.isnan(rating):
        print("User actually not similar", user, movie)
        return np.average([user_movies[user][movie] for movie in user_movies[user]])
    return rating


def dumb(user):
    return np.average([user_movies[user][movie] for movie in user_movies[user]])

def rmse(x,y):
    z = np.square([a-b for a,b in zip(x,y)])
    return math.sqrt(np.sum(z)/len(z))


train_path = '../Data/train.csv'
data = pd.read_csv(train_path)


unique_movies = sorted(list(set(data['MovieID'])))
unique_users = sorted(list(set(data['UserID'])))


user_movies = {}
movies_users = {}

for user in unique_users:
    movies_for_user = data[data.UserID == user][['MovieID', 'Rating']]
    movies_for_user = movies_for_user.set_index('MovieID')['Rating'].T.to_dict()
    user_movies[user] = movies_for_user

for movie in unique_movies:
    users_seeing_movie = list(data[data.MovieID == movie]['UserID'])
    movies_users[movie] = users_seeing_movie


number_of_hash_functions = 60
number_of_bands = 20
threshold = 10/20

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


if os.path.isfile('../Data/similar_users.npy'):
    user_neighbours = np.load('../Data/similar_users_1.npy').item()
else:
    user_neighbours = {}
    for user_index in unique_users:
        print(user_index)
        user_neighbours[user_index] = neighbours_of_user(user_index, matrix, number_of_bands, threshold, user_neighbours)

    np.save('../Data/similar_users_1.npy', user_neighbours)


if os.path.isfile('../Data/cosine_for_similar_users.npy'):
    cosine_numbers = np.load('../Data/cosine_for_similar_users_1.npy').item()
else:
    cosine_numbers = {}
    for user in user_neighbours:
        print(user)
        cosine_numbers[user] = {}
        for similar_user in user_neighbours[user]:
            if similar_user in cosine_numbers:
                cosine_numbers[user][similar_user] = cosine_numbers[similar_user][user]
            else:
                cosine_numbers[user][similar_user] = cosine_similarity(user, similar_user, user_movies)
    np.save('../Data/cosine_for_similar_users_1.npy', cosine_numbers)


test_data = pd.read_csv('../Data/test.csv')


predicted_values = map(lambda x, y: predict_rating(x, y, cosine_numbers, movies_users, user_neighbours, user_movies),
                       test_data['UserID'], test_data['MovieID'])


dumb_values = map(lambda x: dumb(x), test_data['UserID'])


print(rmse(list(predicted_values), list(test_data['Rating'])))
print(rmse(list(dumb_values), list(test_data['Rating'])))
