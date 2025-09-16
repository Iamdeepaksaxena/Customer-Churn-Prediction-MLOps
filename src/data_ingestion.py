import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addFilter(console_handler)

# Data loading
def load_data(data_url:str) ->pd.DataFrame:
    """Load data from CSV file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurs while loading the data: %s', e)
        raise


# Data preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the Data"""
    try:
        df.drop(columns = ['customerID'], axis = 1, inplace = True)
        df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)
        logger.debug('Data preprocessing is completed')
        return df
    except KeyError as e:
        logger.error('Missing columns in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected Error during the preprocessing %s', e)
        raise

# saving the Preprocessed data

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    "Saving the train and test data"
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        # saving the data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and Test data Saved to %s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected errors occurs while saving the data: %s',e)
        raise


def main():
    try:
        test_size = 0.20
        data_path ='Telcom Data.csv'
        df = load_data(data_url= data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error("Data ingestion process is failed: %s",e)
        print(f'Error{e}')

if __name__ =='__main__':
    main()
