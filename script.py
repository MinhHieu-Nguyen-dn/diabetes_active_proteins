import streamlit as st
import pandas as pd
from preprocessing import smiles_to_fp
import pickle
import numpy as np

# Load models
model_AKR1B1 = pickle.load(open("models/model_xgboost_final.pkl", "rb"))
model_SGLT2 = pickle.load(open("models/model_rf_final.pkl", "rb"))


def predict(input_df):
    data_input = input_df["smiles"].apply(smiles_to_fp).tolist()
    proba_AKR1B1 = model_AKR1B1.predict_proba(data_input)
    result_AKR1B1 = model_AKR1B1.predict(data_input)
    proba_SGLT2 = model_SGLT2.predict_proba(data_input)
    result_SGLT2 = model_SGLT2.predict(data_input)
    return result_AKR1B1, np.max(proba_AKR1B1, axis=1), result_SGLT2, np.max(proba_SGLT2, axis=1)


st.set_page_config(page_title="Compounds vs. Diabetes Target", layout="wide")
st.title('Predict: Compounds/Proteins Interactions')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df[['molecule_chembl_id', 'smiles']]
    result_AKR1B1, proba_AKR1B1, result_SGLT2, proba_SGLT2 = predict(input_df)
    input_df['Aldose reductase (AKR1B1)'] = [f'{r} ({p * 100:.2f}%)' for r, p in zip(result_AKR1B1, proba_AKR1B1)]
    input_df['Sodium/glucose cotransporter 2 (SLC5A2, SGLT2)'] = [f'{r} ({p * 100:.2f}%)' for r, p in
                                                                  zip(result_SGLT2, proba_SGLT2)]
    input_df.rename(columns={'molecule_chembl_id': 'ChEMBL ID', 'smiles': 'SMILES'}, inplace=True)

    # Create a color map for the prediction columns
    def color_cells(val):
        color = 'green' if '1' in val else 'red'
        return f'color: {color}; font-weight: bold'


    # Make the header of the table more impressive
    header_properties = [('font-size', '18px'), ('text-align', 'center'), ('font-weight', 'bold')]
    table_styles = [dict(selector="th", props=header_properties)]

    st.table(input_df.style.applymap(color_cells, subset=['Aldose reductase (AKR1B1)',
                                                          'Sodium/glucose cotransporter 2 (SLC5A2, SGLT2)']).set_table_styles(
        table_styles))