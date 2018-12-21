# Final project : Recommending the next best movie" 

**Team Members:** Ashwin Jayaraman, Somendra Tripathi, Akhil Punia

## About the Movie Lens Data Set

Full: 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. Includes tag genome data with 14 million relevance scores across 1,100 tags. Last updated 9/2018. [Link] (http://grouplens.org/datasets/movielens/latest/)

![fig1](fig/fig11.png)
![fig2](fig/fig12.png)

> **Observations:**
- Of the 2,83,228 users using Movies around 69,000 people rated more then 100 movies and 45,912 users have rated less than 10 movies.
- We would want to remove those users who have rated the movies highly from the catalogue as it maybe from a bot account or professional reviewer. Since, we are trying to model the interrests of the larger public it would make sense to take the data of users who haven't rated more than 100 movies.
- Also new users who have rated less than 10 movies, present the classical cold start problem. We should not use them in building our recommender simply because enough data is not available.

![fig3](fig/fig21.png)
![fig4](fig/fig22.png)

> **Observations:**
- We see that the average rating is centred around the 3.5 and the interquartile range is between 3-4. A lot of movies have been highly rated.

![fig5](fig/fig3.png)

> **Observations:**
- In this particular sample of dataset provided from Movielens, we see a sudden resurgence in the number of available ratings beginning from the year 2015.

![fig6](fig/fig4.png)

> **Observations:**
The Popularity of Movies follow the long tail distibution which exemplifies the pgenomenon of rich gets richer. More Popular movies get more ratings.

![fig7](fig/fig5.png)
> **Observations:**
- Majority of the Users use the platform for rating for only single year. It is expected as it is an academic dataset and not a commercial product which monitors its users over the years.
- This provides an interesting challenge in which the older movies will tend to be the most rated as new movies only have a fraction of overall users to receive ratings from.

Code for EDA is available [here](eda.ipynb).

## Business Problem

### Challenges
- Sparsity

- Size of the DataSet

- Outliers

- Model Explainability and User Trust

- Serendipity

## Approaches
- [Approximate Nearest Neighbors (ANN) using Locality Sensitive Hashing (LSH)](ANN.md)
- Factorization Machines
- [Deep Learning Approaches: Autoencoders and Learning Embeddings through Shallow Networks](DeepLearning.md)

## Summary
- Metrics
  - Quantitaive ( Accuracy, RMSE )
  - Qualitative ( Top K Recommendations )

- Why do Autoencoders don't work ?
  - Cannot distinguish between a bad rating and a sparse data
  - Needs large GPU resources to train on the movielens data
  - Future: Might Have to look into Sparse Autoencoders, or frameworks like Amzon DSSTNE
- Embeddings
  - Do a good job at obtaining a latent space representation of movies.
  - can be visualised using TSNE.
