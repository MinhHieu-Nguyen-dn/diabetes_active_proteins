import pandas as pd
from preprocessing import smiles_to_fp
import pickle

# Get input dataframe from file and create fingerprints column
input_df = pd.read_csv('data/test_multi.csv')
print('Received input dataframe!')
input_df = input_df[['molecule_chembl_id', 'smiles']]
input_df["fp"] = input_df["smiles"].apply(smiles_to_fp)
print('Added fingerprints for input dataframe!')

# Aldose reductase (AKR1B1): uniprot_id = P15121
model_AKR1B1 = pickle.load(open("models/model_xgboost_final.pkl", "rb"))
print('Loaded model for AKR1B1')
# Sodium/glucose cotransporter 2 (SLC5A2, SGLT2): uniprot_id = P31639
model_SGLT2 = pickle.load(open("models/model_rf_final.pkl", "rb"))
print('Loaded model for SGLT2')

data_input = input_df.fp.tolist()
proba_AKR1B1 = model_AKR1B1.predict_proba(data_input)
result_AKR1B1 = model_AKR1B1.predict(data_input)
proba_SGLT2 = model_SGLT2.predict_proba(data_input)
result_SGLT2 = model_SGLT2.predict(data_input)

print("result_AKR1B1\n", proba_AKR1B1, result_AKR1B1)
print("result_SGLT2\n", proba_SGLT2, result_SGLT2)
