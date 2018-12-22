# Model 3 : Deep Learning to learn User and Movie Embeddings

### Model Description

The idea here is to implement a simple matrix factorisation model using deep learning framework. We have chosen a simple model we specify an 8 dimensional embedding layer for user items and movies. These layers are intialised to  a set of random values. 
Next step in our model is to take the dot product of these two layers. It thus helps us reconstruct the ratings for every user for every movie.

<p align="center">
<img src="https://user-images.githubusercontent.com/16842872/50356721-169bc600-0579-11e9-9560-8534c3f58aaa.png" width="600" height="300">
</p>


### Description of the Algorithm 
- In the first step we take the average rating across all users and subtract it from the individual ratings. The resultant column vector is chosen as our target variable. During the time of reconstruction, we can simply add this average rating to the reconstructed value to obtain the actual rating.
- Model is trained using Adam Optimiser. It is an optimisation algoirhtm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

## Parameters of the Model
- Movie & User Embedding Size ( k =8 )
- Learning Rate ( lr = 0.01 )
- Loss Function ( Mean Squared Error )
- Validation Set Size ( 0.05 )
- Epochs ( 20 )
- Batch Size ( 5000 ) 

[Code](training_embeddings.ipynb)

## Parameter Tuning and Optimal Model 
- The value of k ( embedding size) is chosen to be 8 after making a tradeoff between training time and model size.
- Higher values of k tend to provide low training error but no imporvement is observed in the validation set which makes us believe that our model maybe **Overfitting**.

<p align="center">
<img src="https://user-images.githubusercontent.com/16842872/50356770-4cd94580-0579-11e9-9f84-69bcaf04e65d.png" width="800" height="400">
</p>

## Model Results and Explainability

Once the model is trained we can explore the learnt embeddings as follows. We know that each movie is represented by a 8 dimensional vector. We can compute the vector simialrity to find out the nearest movie to a querried movie. In the code file, I have shown how we can use euclidean and cosine distance for computing this metric. any value between 1 and 2 show the degree of similarity between two movies and values between 0 and 1 denotes how dissimilar two movies are.

[Code](visualising_embeddings.ipynb)

<p align="center">
<img src="https://user-images.githubusercontent.com/16842872/50356831-a04b9380-0579-11e9-855c-2d5e85b56c89.png" width="900" height="600">
</p>

We see the Mean Absolute Error of the above model decreases with ever increasing epoch but the validation set stops improving after 4 epochs. It indicates that a model which avoids overfitting can be trained very quickly. 

> **Qualitative Results:** 
In general, we see that embeddings have done a good job at identifying the similarities between movies.
- Querrying Star Wars identifies all the movies in Sci-Fi and Fantasy genre. It interestingly also suggests its sequels too.
- The Mask correctly brings out all the comedy movies together. Thus, doing a good job at identifying the genre.
- The Nearest Embeddings for Iron Man fetches all the Marvel movies.
- Inception querries surprisingly bring out all the movies from the same director- Christophr Nolan.
- American Sniper fetches an interesting recommendation 'We were Soldiers!', identifying the theme of war and soldiers.

<img src="https://user-images.githubusercontent.com/16842872/50356832-a04b9380-0579-11e9-8230-e092e01a26fb.png" width="900" height="500">

We have also applied TSNE to visualise theses 8 dimensional movie embeddings in 2D. It provides us with an insight as to how how are different movies clustered in the two dimensional space and provides a window towards model explainability.

```python
from sklearn.manifold import TSNE
tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
embs = tsne.fit_transform(w)
```
<img src="https://user-images.githubusercontent.com/16842872/50356833-a0e42a00-0579-11e9-9bc0-16a2ba957201.png" width="900" height="500">

<img src="https://user-images.githubusercontent.com/16842872/50356834-a0e42a00-0579-11e9-91e1-3385cd4ead5b.png" width="900" height="500">



## Concluding Remarks
Given the effectiveness of our model to correctly cluster movies with relevant genrres and popularity. We feel that this model can be used to successfully generate recommendation for a new set of users for whom we do not have much information for. This approach provides us with a undestnding as to how our model is 

## Requirements
```
Tensorflow: 1.1.0
Keras: 2.1.6
Python version : 3.6
RAM : 16GB
```
