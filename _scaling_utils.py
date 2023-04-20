from sklearn.preprocessing import StandardScaler
from logging_utils import setup_logger
import pandas as pd

def standart_scaler(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    logger = setup_logger()
    logger.debug("standart_scaler function is executing...")
    
    scaler = StandardScaler()
    for col in numeric_cols: 
        logger.info(f"{col} is scaling...")
        df[col] = scaler.fit_transform(df[[col]])
            
    return df
