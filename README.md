"Final project : Recommending the next best movie" 

Team Members: Ashwin Jayaraman, Somendra Tripathi, Akhil Punia

# About the Movie Lens Data Set

![fig1](fig/fig11.png)
![fig2](fig/fig12.png)
![fig3](fig/fig21.png)
![fig4](fig/fig22.png)
![fig5](fig/fig3.png)
![fig6](fig/fig4.png)
![fig7](fig/fig5.png)


### Key Features
- Sparsity
- Size of the DataSet

### Challenges
- Outliers

- Model Explainability and User Trust

- 

# Approaches
- [Approximate Nearest Neighbors (ANN) using Locality Sensitive Hashing (LSH)](ANN.md)
- Factorization Machines
- [Deep Learning Approaches: Autoencoders and Learning Embeddings through Shallow Networks](DeepLearning.md)

# Discussion on Results
- Metrics
  - Quantitaive ( Accuracy, RMSE )
  - Qualitative ( Top K Recommendations )
- 

- Why do Autoencoders don't work ?
  - Cannot distinguish between a bad rating and a sparse data
  - Needs large GPU resources to train on the movielens data
  - Future: Might Have to look into Sparse Autoencoders, or frameworks like Amzon DSSTNE
- Embeddings
  - Do a good job at obtaining a latent space representation of movies.
  - can be visualised using TSNE.
