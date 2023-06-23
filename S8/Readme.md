
# Model
### Following is my network's output from torch summary
![op](images/op.png)
### Parameters
- Batch size - 128
- Optimizer - Stochastic gradient descent (SGD)
- Dropout - 0
- Scheduler - None
- Normalization techniques <br/>

| Technique                  |   Best training accuracy | Best test accuracy  |  L1 factor | number of groups |
|:--------------------------:|:------------------------:|:-------------------:|:----------:|:----------------:|
| L1 + Batch normalization   | 94.19% at 20th Epoch     | 92.01% at 1st epoch |    0.01    |       0          |    
| Layer normalization        | 99.35% at 19th Epoch     | 99.29% at 13th epoch|    0       |       0          |    
| Group normalization        | 99.45% at 19th Epoch     | 99.37% at 15th epoch|    0       |       8          |    

# Files
- [model.py](https://github.com/DimpleB0501/ERA1/blob/main/S8/model.py) file includes GN/LN/BN and takes an argument to decide which normalization to use.
- [Session_8_Assignment_QnA.ipynb](https://github.com/DimpleB0501/ERA1/blob/main/S8/S8.ipynb) is a single notebook file to run all the above 3 models for 20 epochs each.

# Normalization techniques brief explanation
Normalization of inputs in neural network removes the difference in magnitude between different features, in turn aiding learning in DNN. <br/> 

In Batch normalization, the mean and standard deviation are calculated for all inputs within a mini batch. <br/> 

In Layer normalization, dependancy on the batch size is removed. Here mean and standard deviation is calculated across all layers (Intuitively if there are 4 layers, we get four mean and variance values for each input in your dataset.)
<br/> 

Group normalization - is a combination of both batch and layer normalization, it divides the features/channels into groups and normalizes each group separately.

# Analysis 
![grap](images/graph.png)
In layer and group normalization the model converges smoothly. For batch normalization there are jumps in test loss and test accuracy, this displays model instability. We have used batch normalization with L1 regularization (l1_lambda = 0.01).    
# 10 misclassified images for each of the 3 models
##### Network with L1 + BN
![batch](images/batch.png)
##### Network with Layer Normalization
![Layer](images/layer.png)
##### Network with Group Normalization
![group](images/group.png)
