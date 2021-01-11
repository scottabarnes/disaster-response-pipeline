# Disaster Response Pipeline Project

![Intro Pic](screenshots/intro.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
3. [Project Organisation](#project_organisation)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The objective of the project was to build a tool that would classify disaster messages into categories - e.g. "water", "fire" or "food" to aid responses. The dataset used to train the model contains pre-labelled request messages sent during a natural disaster.

The project is divided in the following sections:

1. ETL Pipeline:
- Loads the messages and categories dataset
- Merges the two datasets
- Cleans the data
- Stores in a SQLite database

2. ML Pipeline - builds a model able to classify text message into categories
- Loads data from SQLite Database
- Splits the data into training and test setse
- Builds a text processing and ML pipeline  
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Saves the final model as a pickle file

3. Flask Web App
- Allows user to send request messages to model and receive response in real time

<a name="getting_started"></a>
## Getting started:
1. Running the ETL Pipeline:
      - From the project directory run:
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2. Running the ML Pipeline:
    - From the project directory run `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Starting the Flask Web App:
  - From the project directory run `python /app/run.py`
  - Go to http://127.0.0.1:3000/


<a name="project_organisation"></a>
## Getting started:

Project Organization
------------

    ├── app
    │   ├── run.py------------------------# python file used to run the flask app
    │   └── templates
    │       ├── go.html-------------------# classification result page on app
    │       └── master.html---------------# main page on app
    ├── data
    │   ├── DisasterResponse.db-----------# database used for saving and loading data
    │   ├── disaster_categories.csv-------# input data - message labels  
    │   ├── disaster_messages.csv---------# input data - messages
    │   └── process_data.py---------------# python file used to load and clean data
    ├── screenshots-----------------------# screenshots used in README
    ├── models
    │   └── train_classifier.py-----------# builds and trains model used in web app
