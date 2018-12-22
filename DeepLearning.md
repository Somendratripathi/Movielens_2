# Model 3 : Deep Learning to learn User and Movie Embeddings

### Model Description

The idea here is to implement a simple matrix factorisation model using deep learning framework. We have chosen a simple model we specify an 8 dimensional embedding layer for user items and movies. These layers are intialised to  a set of random values. 
Next step in our model is to take the dot product of these two layers. It thus helps us reconstruct the ratings for every user for every movie.

<img src="https://user-images.githubusercontent.com/16842872/50356721-169bc600-0579-11e9-9560-8534c3f58aaa.png" width="600" height="300">



### Description of the Algorithm 
- In the first step we take the average rating across all users and subtract it from the individual ratings. The resultant column vector is chosen as our target variable. During the time of reconstruction, we can simply add this average rating to the reconstructed value to obtain the actual rating.
- Model is trained using Adam Optimiser.


## Parameters of the Model
- Movie & User Embedding Size ( k =8 )
- Learning Rate ( lr = 0.01 )
- Loss Function ( Mean Squared Error )
- Validation Set Size ( 0.05 )
- Epochs ( 20 )
- Batch Size ( 5000 )

[Code](training_embeddings.ipynb)

## Design Choices to consider


## Parameter Tuning and Optimal Model 
- The value of k is chosen to be te 8 after making a tradeoff between training time and 

<img src="https://user-images.githubusercontent.com/16842872/50356770-4cd94580-0579-11e9-9f84-69bcaf04e65d.png" width="800" height="400">


## Model Results and Explainability

<img src="https://user-images.githubusercontent.com/16842872/50356831-a04b9380-0579-11e9-855c-2d5e85b56c89.png" width="900" height="600">
<img src="https://user-images.githubusercontent.com/16842872/50356832-a04b9380-0579-11e9-8230-e092e01a26fb.png" width="900" height="500">
<img src="https://user-images.githubusercontent.com/16842872/50356833-a0e42a00-0579-11e9-9bc0-16a2ba957201.png" width="900" height="500">
<img src="https://user-images.githubusercontent.com/16842872/50356834-a0e42a00-0579-11e9-91e1-3385cd4ead5b.png" width="900" height="500">

[Code](visualising_embeddings.ipynb)

## Requirements
```
Tensorflow: 1.1.0
Keras: 2.1.6
Python version : 3.6
RAM : 16GB
```
