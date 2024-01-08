import pandas as pd
import pickle
from preprocessing import string_to_np_array
from preprocessing import df_to_data_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/processed_P31639.csv', index_col=0)
df['fp'] = df['fp'].apply(string_to_np_array)
data_splits = df_to_data_split(df=df)

model_rf = RandomForestClassifier(
    min_samples_split=5,
    max_depth=None,
    n_estimators=300
)
model_rf.fit(data_splits['x_train'], data_splits['y_train'])
pickle.dump(model_rf, open("models/model_rf_final.pkl", "wb"))
