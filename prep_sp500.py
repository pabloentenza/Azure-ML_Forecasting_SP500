
import os
import argparse
import pandas as pd
from azureml.core import Run

# Get_parameter

parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type = str, dest = 'raw_dataset_id' , help = 'raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get run context

run = Run.get_context()

print("Loading data.....")

df = run.input_datasets['raw_data'].to_pandas_dataframe()

df["DATE"] =  pd.to_datetime(df["DATE"])
df['SP500']= pd.to_numeric(df['SP500'], errors='coerce')
df = pd.DataFrame(df)
df = df.dropna()               
               
row_sp500 = len(df)

run.log('Number_rows', row_sp500)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'SP500.csv')
df.to_csv(save_path, index = False, header=True)

# End the run
run.complete()
