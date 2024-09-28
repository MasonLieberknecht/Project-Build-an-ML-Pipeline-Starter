#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column.
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def ensure_consistent_dtypes(df):
    """
    Ensure that all columns in a DataFrame have consistent data types that are compatible with MLflow DataType.
    Converts all object columns to strings and ensures numerical columns are either int or float.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce invalid parsing to NaN to maintain numeric types
        else:
            raise ValueError(f"Unsupported data type for column: {col} with type: {df[col].dtype}")
    return df


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config['random_state'] = args.random_seed

    # Use run.use_artifact(...).file() to get the train and validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    ######################################
    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    sk_pipe.fit(X_train, y_train)
    ######################################

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    ######################################
    # Ensure all columns in X_val have consistent data types
    X_val = ensure_consistent_dtypes(X_val)

    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    signature = mlflow.models.infer_signature(X_val, y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        path="random_forest_dir",
        signature=signature,
        input_example=X_train.iloc[:5]
    )
    ######################################

    # Upload the model we just exported to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type='model_export',
        description='Trained random forest artifact',
        metadata=rf_config
    )
    artifact.add_dir('random_forest_dir')
    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    ######################################
    # Here we save variable r_squared under the "r2" key
    run.summary['r2'] = r_squared
    # Now save the variable mae under the key "mae".
    run.summary['mae'] = mae
    ######################################

    # Upload to W&B the feature importance visualization
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


# The remaining code for plot_feature_importance and get_inference_pipeline functions
# and the __main__ block remains the same.


