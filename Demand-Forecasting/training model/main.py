import numpy as np
import pandas as pd
import os
#
from datasetsforecast.hierarchical import HierarchicalData
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import  BottomUp, TopDown, MiddleOut, MinTrace, ERM
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
from datetime import datetime
import pandas as pd
from ai_core_sdk.models import Metric, MetricTag, MetricCustomInfo, MetricLabel
#
from ai_core_sdk.tracking import Tracking
aic_connection = Tracking()
#
# Variables
DATA_PATH_TRAIN = '/app/data/train.csv'
MODEL_PATH = '/app/model/model.pkl'
#
# Load Datasets
#
dfx = pd.read_csv(DATA_PATH_TRAIN)
#
# # Metric Logging: Basic
aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "N_observations", value= float(dfx.shape[0]), timestamp=datetime.utcnow()),
    ]
)
#
# Partition into Train and test dataset
#
horizon = 7 
x_test = dfx.groupby('unique_id').tail(horizon)
x_train = dfx.drop(x_test.index)
x_test = x_test.set_index('unique_id')
x_train = x_train.set_index('unique_id')
# Ensure the target column 'y' is numeric
x_train['y'] = pd.to_numeric(x_train['y'], errors='coerce')
# Drop or handle NaN values if any were introduced
x_train = x_train.dropna(subset=['y'])
#
#Final Model
#
# Compute base auto-ARIMA predictions
fcst = StatsForecast(df = x_train, models=[AutoARIMA(season_length= 7)], freq='D', n_jobs=-1)
x_hat = fcst.forecast(h = horizon)
xmat = pd.merge(left = x_test, right = x_hat, on = ['ds', 'unique_id'])
#scoring over test data
rmse = rmse(xmat['y'], xmat['AutoARIMA'])
# Metric Logging: Attaching to metrics to generated model
aic_connection.log_metrics(
    metrics = [
        Metric(
            name= "Test data RMSE",
            value= float(rmse),
            timestamp=datetime.utcnow(),
            labels= [
                MetricLabel(name="metrics.ai.sap.com/Artifact.name", value="demand-forecasting")
            ]
        )
    ]
)
#
# Save model
import pickle
pickle.dump(fcst, open(MODEL_PATH, 'wb'))
#
#Add tags
aic_connection.set_tags(
    tags= [
        MetricTag(name="Validation Set", value= "7 last days of dataset"), # your custom name and value
        MetricTag(name="Metrics", value= "RMSE"),
    ]
)
