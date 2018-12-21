import surprise
import pandas as pd
import time
import pickle


train_path = 'Data/train.csv'
df = pd.read_csv(train_path)
reader = surprise.Reader(rating_scale=(1, 5))
data = surprise.Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'msd'}

model = surprise.KNNBasic(sim_options=sim_options)
t1 = time.time()
model.fit(trainset)
t2 = time.time()

print(t2 - t1)
with open("Data/time_surprise.txt", "wb") as f:
    pickle.dump([t2-t1], f)

exit(0)
test_path = 'Data/test.csv'
df = pd.read_csv(test_path)
test_data = surprise.Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)\
    .build_full_trainset().build_testset()
predictions = model.test(test_data)

print(surprise.accuracy.rmse(predictions))

## 0.9969192445535927