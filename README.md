# Disaster_response_pipeline
In this project I have built a model to classify disaster messages  during disasters
the data set to this project is provided by Figure Eight. 

###File Description

1)App folder including the templates folder and "run.py" for the web application

2)Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.

3)Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.

4)README file

5)Preparation folder containing 6 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)

###How to run

1)Run the following commands in the project's root directory to set up your database and model.

	a)To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv 	data/disaster_categories.csv data/DisasterResponse.db

	b)To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db 		models/classifier.pkl

2)Run the following command in the app's directory to run your web app. python run.py
