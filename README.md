
This is final homework for RS School Machine Learning course.
This demo uses Forest cover type data set from kaggle.com.
The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. 
Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. 
There are seven types:
* 1 - Spruce/Fir
* 2 - Lodgepole Pine
* 3 - Ponderosa Pine
* 4 - Cottonwood/Willow
* 5 - Aspen
* 6 - Douglas-fir
* 7 - Krummholz 

## Usage
This package allows you to train model for detecting the forest cover type prediction.


Clone this repository to your machine.
Download Heart Disease dataset, save csv locally (default path is data/heart.csv in repository's root).
Make sure Python 3.9 and Poetry are installed on your machine (I use Poetry 1.1.11).
Install the project dependencies (run this and following commands in a terminal, from the root of a cloned repository):
poetry install --no-dev
Run train with the following command:
poetry run train -d <path to csv with data> -s <path to save trained model>
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:

poetry run train --help
Run MLflow UI to see the information about experiments you conducted:
poetry run mlflow ui