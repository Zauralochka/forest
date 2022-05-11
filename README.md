
This is final homework for RS School Machine Learning course.
This demo uses [Forest cover type data set] (https://www.kaggle.com/competitions/forest-cover-type-prediction/data) from kaggle.com.
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

1.   Clone this repository to your machine.
2. Download dataset (see link above), save csv locally (default path is data/train.csv and data/test.csv in repository's root).
3. Make sure Python 3.9 and Poetry are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (run this and following commands in a terminal, from the root of a cloned repository): \

```poetry install --no-dev```

5. Run train with the following command:\

```poetry run train -d <path to csv with data> -t <path to csv with data> -s <path to save trained model>```

6. You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:\

```poetry run train --help```\

This step didn't implemented yet. Training will run on the predifined group of parameters.
7. Run MLflow UI to see the information about experiments you conducted:

```poetry run mlflow ui```

![You will get the simular result as on the this screenshot](https://github.com/Zauralochka/forest/blob/main/images/screenshot_mlflow.png)

## Development
The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment: \

```poetry install```

Now you can use developer instruments, e.g. pytest:

```poetry run pytest```

The test module wasn't implemented yet.

More conveniently, to run all sessions of testing and formatting in a single command, install and use nox:

```nox [-r]```

Format your code with black by using either nox or poetry:

```nox -[r]s black```
```poetry run black src tests noxfile.py```