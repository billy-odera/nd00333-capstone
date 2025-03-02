# Udacity Azure ML Nanodegree Capstone Project - Predicting the Survival of Titanic Passengers
This project uses [Kaggle Titanic Prediction Dataset](https://www.kaggle.com/c/titanic/data) in azure workspace to train models using the different tools and deploy the best machine learning model as a web service using python sdk.

## Project Pipeline
![Pipeline](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/pipeline.jpg)

The sinking of the Titanic is one of the most infamous shipwrecks in history.On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. This resulted in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this project, I build a classification model to predictive whether the passenger will survive or not.

## Project Set Up and Installation
To run this project, you will need an active account on Kaggle. From Kaggle, search for Titanic, enter the competition and download the dataset. Import the data into your Azure ML Studio

## Dataset

### Overview
We use this dataset from [Kaggle Titanic Prediction Dataset](https://www.kaggle.com/c/titanic/data). The data dictionary is as below:

<ul>
  <li>survival (Survival	0 = No, 1 = Yes)</li>
  <li>pclass - Ticket class	(1 = 1st, 2 = 2nd, 3 = 3rd)</li>
  <li>sex	- Gender</li>
  <li>Age	- Age in years	</li>
  <li>sibsp	- # of siblings / spouses aboard the Titanic	</li>
  <li>parch	- # of parents / children aboard the Titanic</li>	
  <li>ticket - 	Ticket number</li>	
  <li>fare	- Passenger fare</li>	
  <li>cabin	Cabin number	</li>
  <li>embarked- Port of Embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)</li>
</ul>

### Task
The task for this project is to train models to classify whether a passenger will survive the titanic shipwreck The 'Survived' column in the dataset is 1 when the passenger survived from the shipwrecks and 0 when he/she didnt survive.  


### Access
Download the data from Kaggle,once the data is downloaded, register it to the Azure ML workspace. There is some  data cleaning done before the model is trained.

## Automated ML
The [automl](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/automl.ipynb) notebook will run you through the steps of configuring and running the AutoML experiment. We do a Classification task on the 'Survived' column from the titanic dataset. We also set the primary metric to 'accuracy' with auto featurization, and a timeout set at 30 minutes.

### Results
<p>The below snapshot shows different models generated by automl feature. </p>

![AutoML RunDetails Notebook](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/c.jpg)

![AutoML RunDetails](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/e.jpg)

<p>The voting ensebmble model was the best with an accuracy of 82.9% . </p>

![Best Model](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/f.jpg)

![best model complete](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/g.jpg)


## Hyperparameter Tuning
The [hyperparamter tuning](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/hyperparameter_tuning.ipynb) notebook will run you through the steps for the Hyperdrive run. I have chosen the <b>Random Forest</b> model. Random Forest models generally provide a high accuracy because the are ensemble models(bagging).

For the hyperparameter tuning of this model, we will be tuning four different paramaters for the forest using a random parameter sampling:
<ul>
  <li>n_estimators: The number of trees in the Random forrest</li> 
  <li>max_depth: The maximum depth of the trees in the forrest</li>
  <li>min_samples_split: The minimum number of samples required to split an internal node</li>
  <li>min_samples_leaf: The minimum number of samples required to be at a leaf node</li>
</ul>


### Results
<p>We got an accuracy of 86.8% </p>

![hyperdrivel](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/f.jpg)

![hyperdrive run](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/k.jpg)

## Model Deployment
<p>I deployed the model(Voting Ensemble) generated by automl  model generated by automl. I registered and deployed this model as a web service using ACI (Azure Container Instance).The sample data feeded to the deployed model as a web service request as shown below</p>
   
![Web Service 1](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/h.jpg)

![Web Service 2](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/i.jpg)

![Web Service 3](https://github.com/billy-odera/nd00333-capstone/blob/master/starter_file/screenshot/j.jpg)

## Screen Recording
<p> Below is the link to the video recording</p>

[![Azure ML Capstone](https://img.youtube.com/vi/m6WDCijyR2Q/0.jpg)](https://www.youtube.com/watch?v=m6WDCijyR2Q "capstone project")

## Standout Suggestions
<p>In this deployment I have enabled <b>Application Insight</b> which helps in logging and monitoring of web service.</p>

## Future work
<ul>
  <li> Do more work on data cleaning eg the name column and feature engineering to get more valuable columns.</li>
  <li> Work on conversion of registered model to ONNX format.</li>
  <li> Audit the models for overfitting and what measures can be put in place to deal with imbalanced dataset. </li>
</ul>

