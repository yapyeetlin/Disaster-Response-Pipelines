# Disaster Response Pipeline Project

## Instructions
0. Install the required packages *(optional if done once)*<br>

    - From requirements.txt file with the command:<br>
        `pip install -r requirements.txt`

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:<br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:<br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the workspace directory to run your web app.<br>
    `cd app` -> `python run.py`

3. Go to http://localhost:3001/


## About
This project is made as part of course fulfillment for Udacity's Data Scientist Nanodegree Program in collaboration with Bosch AI Talent Accelerator Scholarship Program.


## Workspace Description
~~~
        disaster_response_pipeline
        |-- app                            
            |-- templates                   //html files for the web application
                |-- go.html
                |-- master.html
            |-- run.py                      //Flask file to run the web application
        |-- data
            |-- disaster_messages.csv       // text data for disaster messages
            |-- disaster_categories.csv     // category data for disaster messages
            |-- DisasterResponse.db         // output of the ETL pipeline
            |-- process_data.py             // script for ETL pipeline
        |-- models
            |-- classifier.pkl              //classifier model serialized in pickle
            |-- train_classifier.py         //script for ML pipeline
        |-- requirements.txt  
        |-- README
        |-- Screenshots
        |-- .archives                       // misc. files for data science activities
~~~
