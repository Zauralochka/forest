from joblib import dump
from pathlib import Path
from shutil import copy2 as copy

import click
import mlflow
import mlflow.sklearn
# import matplotlib.pyplot as plt  # TODO: draw confusion matrix
import pandas as pd

from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix  # TODO: draw confusion matrix

from .data import get_data, create_submission
from .pipeline import create_pipeline, ModelType


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="train.csv",  # default="data/train.csv"
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-t",
    "--test-path",
    default="test.csv",  # default="data/train.csv"
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="model.joblib",  # default="data/model.joblib"
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-submission-path",
    default="submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=17,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.1,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    dataset_path: Path,
    test_path: Path,
    save_model_path: Path,
    save_submission_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    # raise AssertionError('OLOLO!')
    features_train, features_valid, target_train, target_valid = get_data(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    test_df = pd.read_csv(test_path, index_col='Id')
    model_type = ModelType.LOGREG
    lgbm_params = {}
    with mlflow.start_run():
        for model_type in ModelType:
            model_type = model_type.value
            click.echo(f"Model: {model_type}")
            pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state,
                                       model_type, **lgbm_params)
            pipeline.fit(features_train, target_train)
            predictions = pipeline.predict(features_valid)
            accuracy = accuracy_score(target_valid, predictions)

            
            submission_predict = pipeline.predict(test_df)
            submission_stem = save_submission_path.stem
            submission_suffix = save_submission_path.suffix
            submission_path = save_submission_path.with_name(
                f"{submission_stem}.{model_type}{submission_suffix}"
            )
            create_submission(submission_predict, submission_path)
            copy(submission_path, save_submission_path)

            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("maxb_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
            mlflow.log_metric("accuracy", accuracy)

            click.echo(f"Accuracy: {accuracy}")

            model_stem = save_model_path.stem
            model_suffix = save_model_path.suffix
            model_path = save_model_path.with_name(
                f"{model_stem}.{model_type}{model_suffix}"
            )
            dump(pipeline, model_path)
            copy(model_path, save_model_path)

            click.echo(f"Model is saved to {model_path}")
            if model_type in (ModelType.FOREST.value, ModelType.LGBM_A.value,
                              ModelType.LGBM_B.value, ModelType.LGBM_C.value):
                click.echo("Features:")
                feature_importances = (
                    pipeline[-1].best_estimator_.feature_importances_
                    if isinstance(pipeline[-1], BaseSearchCV)
                    else pipeline[-1].feature_importances_
                )
                click.echo(
                    pd.DataFrame(
                        feature_importances,
                        index=features_train.columns, columns=['Importance']
                    ).sort_values(
                        by='Importance', ascending=False
                    )[:10]
                )
            if isinstance(pipeline[-1], BaseSearchCV):
                # Update params from grid search and show results
                lgbm_params.update(pipeline[-1].best_params_)
                click.echo(f"Best score: {pipeline[-1].best_score_}\nParams:")
                for k, v in pipeline[-1].best_params_.items():
                    click.echo(f"{k} = {v}")
            click.echo("")
        click.echo(f"Last model is saved to {save_model_path}")
