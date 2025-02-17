### Disaster Response Pipeline Project

# Overview

This project develops a web application that classifies disaster-related messages into relevant categories to support 
emergency response efforts. Using a machine learning model trained on a dataset of disaster messages, the application automates classification to enhance disaster relief operations and ensure efficient resource allocation.

# Setup instructions
	1) # Install libraries
	Installation of python packages

pip install pandas numpy sqlalchemy nltk scikit-learn flask plotly wordcloud joblib

	2) # Prepare the Database and Train the Model
Run the following scripts in the project's root directory to process data and train the model:

- Run ETL Pipeline
This script cleans the data and stores it in a SQLite database.
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- Run ML Pipeline
This script trains the machine learning model and saves it as a pickle file.

python models/train_classifier.py models/classifier.pkl

	3) # Launch the Web Application

Start the web app by running the following command in the project's root directory:
python run.py
Then, open your browser and access the app at:

- https://lawknodlo5.prod.udacity-student-workspaces.com/
# Project Structure

project-root/
├── app/
│   ├── run.py                # Main Flask application
│   ├── templates/
│   │   ├── master.html       # Main web page template
│   │   ├── go.html           # Classification result template
├── data/
│   ├── process_data.py       # ETL pipeline script
│   ├── disaster_messages.csv # Raw dataset - messages
│   ├── disaster_categories.csv # Raw dataset - categories
│   ├── DisasterResponse.db   # Processed SQLite database
├── models/
│   ├── train_classifier.py   # Machine learning pipeline script
│   ├── classifier.pkl        # Trained classification model
├── README.md                 # Project documentation

# Web Application Features

The web app provides the following data visualizations to enhance disaster response insights:

- Message Genres Distribution – A bar chart showing message distribution across genres (e.g., direct, social, news).

- Category Counts – A bar chart displaying the number of messages per category (e.g., related, request, offer).

- Top 10 Disaster Categories – A visualization highlighting the most common disaster-related message categories.

- Most Used Words – A word cloud and bar chart showcasing the most frequent words in the dataset.

# Machine Learning Pipeline

The machine learning pipeline leverages GridSearchCV for hyperparameter tuning, ensuring optimal model performance. This approach helps in selecting the best parameters to improve accuracy and efficiency in classifying disaster messages.

The trained model is then saved as a pickle file (classifier.pkl), allowing seamless deployment within the web application.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


# Contribution
Contributions are welcome! If you'd like to improve this project, follow these steps:

Fork the Repository – Click the Fork button at the top right of this repository.
Clone Your Fork – Use the following command to clone your forked repository:

Create a New Branch – Work on a separate branch for your changes:

Make Changes & Commit – After making your changes, commit them with a descriptive message:


Push & Submit a Pull Request – Push your branch and open a pull request:

Guidelines for Contribution
✔ Follow best coding practices and maintain clean, readable code.
✔ Document your changes properly.
✔ Ensure that your updates do not break existing functionality.

Let's collaborate to make this project even better! 


