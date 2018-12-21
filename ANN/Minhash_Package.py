from datasketch import MinHash, MinHashLSH
import findspark

findspark.init()
from pyspark import SparkContext,SparkConf

config = SparkConf().setAppName("LSH")

sc = SparkContext(conf=config)

train_path = '../Data/train.csv'


data = sc.textFile(train_path)
header = data.first()


#
# set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
#             'estimating', 'the', 'similarity', 'between', 'datasets'])
# set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
#             'estimating', 'the', 'similarity', 'between', 'documents'])
# set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
#             'estimating', 'the', 'similarity', 'between', 'documents'])
#
# m1 = MinHash(num_perm=128)
# m2 = MinHash(num_perm=128)
# m3 = MinHash(num_perm=128)
# for d in set1:
#     print(d)
#     m1.update(d.encode('utf8'))
# for d in set2:
#     m2.update(d.encode('utf8'))
# for d in set3:
#     m3.update(d.encode('utf8'))
#
#
# exit(0)
'''
For us, the movies the user has seen is the items

So we need to identify which movies have been seen by the user

'''


data = data.filter(lambda x: x != header).map(lambda x: x.split(",")[:-1])\
    .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

user_movies = data.map(lambda x: (x[0], x[1])).groupByKey()\
    .map(lambda x: (x[0], set(x[1]))).sortByKey().collectAsMap()

lsh = MinHashLSH(threshold=0.3, num_perm=60)

m_list = {}

for user in user_movies:
    m = MinHash(num_perm=60)
    for item in user_movies[user]:
        m.update(str(item).encode('utf8'))
    m_list[user] = m
    lsh.insert(user, m)

for user in user_movies:
    if user == 1:
        print(user, lsh.query(m_list[user]))

exit(0)




# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
result = lsh.query(m1)
print("Approximate neighbours with Jaccard similarity > 0.5", result)