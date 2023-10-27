

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


from sklearn.metrics import mean_squared_error
from azureml.core import Run, Model
import argparse
import joblib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union

import warnings
warnings.filterwarnings("ignore")




def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    
    results = []
    
    for order in order_list:
        try: 
            model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, d: int, order: tuple) -> list:
    
    total_len = train_len + horizon
    pred_ARMA = []
    
    for i in range(train_len, total_len, window):
        model = SARIMAX(endog=df, order=(order[0],d,order[1]))
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oos_pred = predictions.predicted_mean.iloc[-window:]
        pred_ARMA.extend(oos_pred)
        
    return pred_ARMA


parser = argparse.ArgumentParser()
parser.add_argument("--training-data",
                    type = str, 
                    dest = 'training_data', 
                    help = 'training data' 
                    )

args = parser.parse_args()
training_data = args.training_data


# Get the experiment run context

run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_data,'SP500.csv')
df = (pd.read_csv(file_path, parse_dates = ["DATE"])
        .set_index("DATE")
     )

shape = df.shape
run.log("DF shape", shape)

d = 0
p_value = adfuller(df)[1]
for i in range(5):
    
    if p_value > 0.05:
        eps_diff = np.diff(df['SP500'], n = i + 1 )
        p_value = adfuller(eps_diff)[1]
        d = d  + 1
        
    else:
        break
print(f"The value of integration d is: {d}")


# Combination of multiple values for p and q from zero to five
n = 5
order_list = [(p,q) for p in range(0,n+1) for q in range(0,n+1)]


limit = np.int(0.2*len(df))
train = df.iloc[:-limit]
test = df.iloc[-limit:]

result_df = optimize_ARIMA(train, order_list, d)
order = result_df.iloc[0,0]

model = SARIMAX(train, order=(order[0],d,order[1]), simple_differencing=False)
model_fit = model.fit(disp=False)



TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1
pred_ARMA = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW,d,order)

df_test  = pd.DataFrame(test)
df_test["pred_fort"] = pred_ARMA

rmse = mean_squared_error(df_test['SP500'], df_test["pred_fort"])


fig, ax = plt.subplots()
ax.plot(df_test['SP500'].iloc[:30], label='Actual')
ax.plot(df_test["pred_fort"].iloc[:30], label='Test')
fig.autofmt_xdate()
plt.tight_layout()
run.log_image('Test', plot = fig)
print('RMSE:', rmse)
run.log('RMSE', np.float(rmse))

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'sp500_model.pkl')
joblib.dump(value=model, filename=model_file)


# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
            model_path = model_file,
            model_name = 'sp500_model',
            tags={'Training context':'Pipeline'},
            properties={'RMSE': np.float(rmse)})

run.complete()

