# Arvato-Project

## Table of Contents

1. [Dependencies](https://github.com/poojapatel26/Arvato-Project#dependencies)
2. [Description](https://github.com/poojapatel26/Arvato-Project#description)
3. [Data files](https://github.com/poojapatel26/Arvato-Project#data-files)
4. [Project Motivation](https://github.com/poojapatel26/Arvato-Project#project-motivation)
5. [File Description](https://github.com/poojapatel26/Arvato-Project#file-description)
6. [Results](https://github.com/poojapatel26/Arvato-Project#results)
7. [Licensing, Authors, Acknowledgements](https://github.com/poojapatel26/Arvato-Project#licensing-authors-acknowledgements)
8. [References](https://github.com/poojapatel26/Arvato-Project#references)
  
## Dependencies

* [Python 3*](https://www.python.org/) 
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [Sciki-Learn](https://scikit-learn.org/stable/)

## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Arvato Bertelsmann.

The Project is divided in the following Sections:

1. Customer Segmentation Report: In this section,the unsupervised learning technique is used to identify few characteristics  for company's existing customers compared to the general population of Germany.

2. Supervised Learning Model: In this section, supervised Learning model is used to investigate mailout_train and mailout_test dataset to predict which individuals are most likely to respond to a mailout campaign.

3. Kaggle Competition: After chosing the best model, the results submitted to [kaggle](http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140) competition.

## Data files

* `azdias`: demographics data for the general population of Germany; 
               891 211 persons (rows) x 366 features.
               
* `customers`: demographics data for customers of a mail-order company; 
                191 652 persons (rows) x 369 features.
                
* `mailout_train`: Demographics data for individuals who were targets of a marketing campaign; 
                   42982 persons and 367 features including response of people.
                   
* `mailout_test`: Demographics data for individuals who were targets of a marketing campaign; 
                  42833 persons and 366 features.

## Project Motivation

The main goal of this project is to characterize the customer segment of the population, and to build a model that will be able to predict customers for Arvato Financial Solutions

## File Description
There are mainly two Notebooks available,

• `Arvato Project Customer Segmentation Report.ipynb` : It includes Data analysis and Unsurvised learning techinques to  compare general population to the company's customers. 

• `Arvato Project ML prediction.ipynb` : It includes Supervised learning techniques to predict which individuals are most likely to respond to a mailout campaign.

And two python files,

• `cleaning.py` : It describes the data preprocessing and cleaning functions of **azdias** and **customers** dataset.

• `ml.py` : It describles the data preprocessing and cleaning functions of **mailout_train** and **mailout_test** dataset and model evaluation functions.

## Results
The main findings of the code can be found at this Customer Segemnetaion Report available here.

## Licensing, Authors, Acknowledgements

  * [Udacity](https://www.udacity.com/) for providing such a Amazing project
  * [Arvato Bertelsmann](https://www.bertelsmann.com/divisions/arvato/#st-1) for providing datasets


## References 

Model Evaluation for ROC Curve Explained

https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

https://www.dataschool.io/roc-curves-and-auc-explained/

https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5



