import pandas as pd
import os
import logging 

# logging configuration

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

url = f"https://drive.google.com/uc?id=16Yiuj61iwTupIZgCx_dvbygIaFTgHOge"

# load data
def load_data(url):
    try:
        df = pd.read_csv(url)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
    
def main():
    df = load_data(url)
    if df is not None:
        data_path = os.path.join('data', 'external')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        save_data(df, 'data/external/uber.csv')

if __name__ == "__main__":
    main()