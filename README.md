# ScaledHome Experiment Management System

This repo illustrates the work done in collaboration with the team leaded by prof. Damla Turgut and prof. Lotzi Boloni during my period abroad at the University of Central Florida (Orlando, FL).

In particular, thanks to the help offered by [Siavash Khodadadeh](https://github.com/siavash-khodadadeh) this part of the project consists of : 
 - a pre-processing module of the data collected from the [ScaledHome Control System](https://github.com/MatteoMendula/ScaledHome_Control_System) 
 - a simulation support model to load new scenarios and run different real-world simulations
 - the implementation of a set of predictive ML models meant to forecast the temperature variation over the simulation period
 - an agent to select among the possibile actions, those that minimize the energy consumption usage
 - 
## Simulate real-world scenarios
The graphs below show as a final result a considerably accurate mapping of real-world temperatures in Milan into the ScaledHome temperature range.
Also, we evaluate the improvement achieved by applying hysteresis techniques at this time. 

![Image of Milan](https://github.com/MatteoMendula/ScaledHome_Experiment_Management_System/blob/master/imgs/milan.png?raw=true)

## Predictive Machine Learning Models

We implemented four different regressors:

 - K-nearest Neighbors
 - Support Vector Machine 
 - Deep Neural Network
 - Long Short-Term Memory

The results acheived are shown in the table below:

| | 300 records| 4000 records| 
| :---: | :---: | :---: |
| **KNN**| 70% and 71%(val.)*| 70% and 71%(val.)* |
| **SVR**| 42% and 47%(val.)*| 22% and 19%(val.)*|
| **DNN**| 79% and 63%(val.)*| 89% and 88%(val.)*|
| **LSTM**| 64% and 27%(val.)*| 87% and 85%(val.)*|
(*) score achieved by training on training set and validation set merged

For each model implemented, we considered both MSE and accuracy.
Regarding the latter, we defined a tolerance range to validate the predicted value, increasing a score variable every time our prediction was inside the tolerance range. 
Since temperature is the main feature we predicted, and the accuracy of the available sensors is 1$^{\circ}$C, we set the tolerance range for each actual target value y<sub>i</sub> to [y<sub>i-1</sub>, y<sub>i+1</sub>]. 


Download the data from 
[here](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/9Z6CKW/6C3C1Y&version=1.2)
and put it into data directory.
