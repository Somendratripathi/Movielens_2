import surprise
import pandas as pd
import time
import os
import pickle


train_path = 'Data/train.csv'
df = pd.read_csv(train_path)
reader = surprise.Reader(rating_scale=(1, 5))
data = surprise.Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine'}
model1 = surprise.KNNBasic(sim_options=sim_options)
model2 = surprise.prediction_algorithms.random_pred.NormalPredictor()
t1 = time.time()
model1.fit(trainset)
model2.fit(trainset)
t2 = time.time()

print(t2 - t1)
if not (os.path.isfile('../Data/time_surprise.txt')):
    with open("../Data/time_surprise.txt", "wb") as f:
        pickle.dump([t2-t1], f)

test_path = 'Data/test.csv'
df = pd.read_csv(test_path)
test_data = surprise.Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)\
    .build_full_trainset().build_testset()
predictions1 = model1.test(test_data)
predictions2 = model2.test(test_data)

print(surprise.accuracy.rmse(predictions1))
print(surprise.accuracy.rmse(predictions2))

rmse_values = [surprise.accuracy.rmse(predictions1), surprise.accuracy.rmse(predictions2)]
if not (os.path.isfile('../Data/knn_rmse.txt')):
    with open("../Data/knn_rmse.txt", "wb") as f:
        pickle.dump(rmse_values, f)
