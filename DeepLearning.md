# Model 3 : Deep Learning Method to learn User and Movie Embeddings

### Model Description
- 

### Description of the Algorithm 
- 

### Handling of Edge Cases
- When a user is present in the test dataset but we do not have any information about that user in the training dataset, the model will return a prediction of 3 which is the middle value of the rating. A possible way to handle would be to ensure that the train test split is not done randomly and certain datapoints from each user is included in both the training and testing data. 
- If a user does not have any neighbours, we then return the mean rating of the user. Since there is so little information about these users, it would be better to keep them in the traininig data only until we start finding neighbours for this particular user. 

Both the above cases can be implemented when the model is run in production and needs to be updated regularly. That would allow us to learn about these users as they start exploring the platform. 

The total number of such cases in the test data was 2% and should not reflect on the accuracy of the model. 

## Parameters of the Model
-


## Design Choices to consider


## Parameter Tuning and optimal Model 



## Requirements
```
Tensoflow: 1.1.0
Keras: 2.1.6
Python version : 3.6
RAM : 16GB
Free Space : 20GB
```
