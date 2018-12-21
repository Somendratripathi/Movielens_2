import numpy as np


def convert_matrix(data, user, item, rating):
    matrix = data.pivot(index=user, columns=item, values=rating).as_matrix()
    return matrix


def get_user_mean(matrix):
    return np.nanmean(matrix, axis=1)


def centralize_matrix(matrix):
    user_mean = get_user_mean(matrix)
    centered_matrix = matrix - user_mean[:, np.newaxis]
    return centered_matrix


def cosine_matrix(matrix):
    where_are_nan = np.isnan(matrix)
    new_matrix = np.copy(matrix)
    new_matrix[where_are_nan] = 0
    squares = np.square(new_matrix)
    num = new_matrix.T.dot(new_matrix)
    one_matrix = np.copy(matrix)
    one_matrix[~np.isnan(one_matrix)] = 0
    one_matrix[np.isnan(one_matrix)] = 1
    one_matrix = 1 - one_matrix
    temp_matrix = squares.T.dot(one_matrix)
    den_matrix = temp_matrix.T * temp_matrix
    den = np.sqrt(den_matrix)
    output = np.divide(num, den)
    return output


def get_nearest_neighbors(matrix, k):
    t = np.argsort(-1 * matrix)
    return t[:, 1:1+k]


def predict(ratings, similarity, ranking, user, item):
    user_ratings_for_similar_items = ratings[user-1, ranking[item-1]]
    similarity_for_similar_items = similarity[item-1, ranking[item-1]]
    value = np.dot(user_ratings_for_similar_items, similarity_for_similar_items)/sum(
        np.absolute(similarity_for_similar_items))
    return value
