import pandas as pd
import pickle
from preprocessing import string_to_np_array
from preprocessing import df_to_data_split
from xgboost import XGBClassifier

df = pd.read_csv('data/processed_P15121.csv', index_col=0)
df['fp'] = df['fp'].apply(string_to_np_array)
data_splits = df_to_data_split(df=df)

model_xgboost = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200
)
model_xgboost.fit(data_splits['x_train'], data_splits['y_train'])
pickle.dump(model_xgboost, open("models/model_xgboost_final.pkl", "wb"))
