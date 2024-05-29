import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.metrics import mean_squared_error as mse
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from ai_core_sdk.models import Metric, MetricLabel, MetricTag
from ai_core_sdk.tracking import Tracking
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables
PREPROCESSED_DATA_PATH = '/app/data/preprocessed/preprocessed_data.csv'
MODEL_PATH = '/app/model/model.pkl'
aic_connection = Tracking()

try:
    # Load Preprocessed Data
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        raise FileNotFoundError(f"{PREPROCESSED_DATA_PATH} not found")
    dfx = pd.read_csv(PREPROCESSED_DATA_PATH, parse_dates=['ds'])
    logger.info("Loaded preprocessed data")

    # Partition into Train and Test datasets
    horizon = 7
    x_test = dfx.groupby('unique_id').tail(horizon)
    x_train = dfx.drop(x_test.index)
    x_test = x_test.set_index('unique_id')
    x_train = x_train.set_index('unique_id')
    x_train['y'] = pd.to_numeric(x_train['y'], errors='coerce')
    x_train = x_train.dropna(subset=['y'])

    # Model Training
    fcst = StatsForecast(df=x_train, models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    x_hat = fcst.forecast(h=horizon)
    xmat = pd.merge(left=x_test, right=x_hat, on=['ds', 'unique_id'])
    
    # Scoring
    rmse = np.sqrt(mse(xmat['y'], xmat['AutoARIMA']))

    # Metric Logging
    aic_connection.log_metrics(
        metrics=[
            Metric(
                name="Test data RMSE",
                value=float(rmse),
                timestamp=datetime.utcnow(),
                labels=[
                    MetricLabel(name="metrics.ai.sap.com/Artifact.name", value="demand-forecasting")
                ]
            )
        ]
    )

    # Save Model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(fcst, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Add tags
    aic_connection.set_tags(
        tags=[
            MetricTag(name="Validation Set", value="7 last days"),
            MetricTag(name="Metrics", value="RMSE"),
        ]
    )
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise
