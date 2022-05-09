from enum import Enum
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


class ModelType(Enum):
    LOGREG = 'logreg'
    FOREST = 'forest'
    LGBM_A = 'lgbm-a'  # 1st stage of hyper-param tuning: tuning model complexity
    LGBM_B = 'lgbm-b'  # 2nd stage of hyper-param tuning: convergence
    LGBM_C = 'lgbm-c'  # final model
    model_type: ModelType,
    **kwargs
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(
            (
                "scaler", StandardScaler()
            )
        )

    if model_type in (ModelType.LOGREG, ModelType.LOGREG.value):
        # Baseline regression (logistic regression)
        pipeline_steps.append(
            (
                "classifier_logreg",
                LogisticRegression(
                    solver='lbfgs',
                    max_iter=max_iter,
                    random_state=random_state,
                    n_jobs=4,
                    multi_class='multinomial',
                    C=logreg_C
                )
            )
        )
    elif model_type in (ModelType.FOREST, ModelType.FOREST.value):
        # Baseline classifier (random forest)
        pipeline_steps.append(
            (
                "classifier_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=4
                )
            )
        )
    elif model_type in (ModelType.LGBM_A, ModelType.LGBM_A.value):
        # Boosting classifier (model complexity)
        pipeline_steps.append(
            (
                "grid_search_complexity",
                GridSearchCV(
                    estimator=LGBMClassifier(random_state=random_state),
                    param_grid={
                        'num_leaves': [7, 15, 31, 63],
                        'max_depth': [3, 4, 5, 6, -1]
                    },
                    cv=5,
                    verbose=1,
                    n_jobs=4
                )
            )
        )

    elif model_type in (ModelType.LGBM_B, ModelType.LGBM_B.value):
        # Boosting classifier (model convergence)
        pipeline_steps.append(
            (
                "grid_search_convergence",
                GridSearchCV(
                    estimator=LGBMClassifier(
                        random_state=random_state,
                        # max_depth=-1,
                        # num_leaves=63,
                        n_estimators=max_iter,
                        n_jobs=1,
                        **kwargs
                    ),
                    param_grid={
                        'learning_rate': np.logspace(-3, 0, 10)
                    },
                    cv=5,
                    verbose=1,
                    n_jobs=4
                )
            )
        )
    elif model_type in (ModelType.LGBM_C, ModelType.LGBM_C.value):
        # Final model
        pipeline_steps.append(
            (
                "classifier_lgbm_final",
                LGBMClassifier(
                    n_estimators=200,
                    # num_leaves=63,
                    # learning_rate=0.2,
                    # max_depth=-1,
                    n_jobs=4,
                    **kwargs
                )
            )
        )
    else:
        raise AssertionError(f"wrong model type {model_type}, must be one of {ModelType}")

    return Pipeline(steps=pipeline_steps)