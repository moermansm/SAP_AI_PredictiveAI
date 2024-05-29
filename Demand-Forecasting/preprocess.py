import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from ai_core_sdk.tracking import Tracking
from ai_core_sdk.models import Metric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables
DATA_PATH_CALENDAR = '/app/data/calendar/calendar.csv'
DATA_PATH_TRAIN = '/app/data/training/sales_train_evaluation.csv'
PREPROCESSED_DATA_PATH = '/app/data/preprocessed/preprocessed_data.csv'
aic_connection = Tracking()

try:
    # Verify if data files exist
    if not os.path.exists(DATA_PATH_CALENDAR):
        raise FileNotFoundError(f"{DATA_PATH_CALENDAR} not found")
    if not os.path.exists(DATA_PATH_TRAIN):
        raise FileNotFoundError(f"{DATA_PATH_TRAIN} not found")
    logger.info("Data files found")

    # Load Datasets
    calendar_df = pd.read_csv(DATA_PATH_CALENDAR, parse_dates=['date'])
    logger.info("Loaded calendar.csv")
    calendar_df = calendar_df.loc[:, ['date', 'wm_yr_wk', 'd']]
    df = pd.read_csv(DATA_PATH_TRAIN)
    logger.info("Loaded sales_train_evaluation.csv")

    df = df.loc[df.item_id == 'FOODS_3_819']
    df_T = df.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    df_T.drop(columns=['id'], inplace=True)

    # Data Preparation
    sales_df = df_T.merge(calendar_df, left_on='variable', right_on='d', how='left')
    sales_df.rename(columns={'value': 'sales_qty'}, inplace=True)
    df = sales_df.loc[sales_df.date >= '2014-01-01', ['date', 'store_id', 'sales_qty']]
    df['state_id'] = df['store_id'].str[:2]
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'] + pd.DateOffset(years=8)

    # Create long format matrices
    df_ind = df.groupby(['date', 'store_id'])[['sales_qty']].sum().reset_index()
    df_ind.columns = ['ds', 'unique_id', 'sales']
    df_sta = df.groupby(['date', 'state_id'])[['sales_qty']].sum().reset_index()
    df_sta.columns = ['ds', 'unique_id', 'sales']
    df_tot = df.groupby(['date'])[['sales_qty']].sum().reset_index()
    df_tot['unique_id'] = 'Total'
    df_tot.columns = ['ds', 'sales', 'unique_id']
    dfx = pd.concat([df_ind, df_sta, df_tot], axis=0)
    dfx.columns = ['ds', 'unique_id', 'y']
    dfx['ds'] = pd.to_datetime(dfx['ds'])

    # Save preprocessed data
    dfx.to_csv(PREPROCESSED_DATA_PATH, index=False)
    logger.info(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")

    # Metric Logging: Basic
    aic_connection.log_metrics(
        metrics=[
            Metric(
                name="N_observations", value=float(dfx.shape[0]), timestamp=datetime.utcnow()
            ),
        ]
    )
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise
