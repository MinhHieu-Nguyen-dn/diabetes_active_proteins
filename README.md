# Predict compounds interaction with diabetes target proteins: SGLT2 and AR.  
Try at: https://compounds-active-diabetes-targets-hieunguyen-datum.streamlit.app/  
*(You can choose "test....csv" file from the `data` folder for testing purpose of this product).*

[Drugs/Proteins Discovery] Predict if an input compound activates **Sodium-Glucose Cotransporter 2 (SGLT2)** and **Aldose Reductase (AR)** as an inhibitor to decrease/block activity of these proteins.  
These 2 proteins are diabetes target, which means they might lead to/cause diabetes if working normally.  
To lower the activity of these 2 proteins, compounds that are predicted "active" can be considered to be chosen as an ingredient in anti-diabetic medications.  

## Outcome of the project:
- User input: A compound or a list of compounds (format: SMILES) - CSV upload file.  
- Output: Result with active/non-active labels corresponding to the interaction of each compound to SGLT2 and AR.
![image](https://github.com/MinhHieu-Nguyen-dn/diabetes_active_proteins/assets/72367686/34f91303-4d5a-4b3b-b3e1-cb82ad8353ba)


## Dependencies and configure settings:
- Virtual environment with Conda and Python 3.8.18:    
`conda create -n diabetes_protein python=3.8`  
`conda activate diabetes_protein`
- Clone this repo (with terminal in your working directory):  
`git clone https://github.com/MinhHieu-Nguyen-dn/diabetes_active_proteins.git`
- Direct to the folder of repo:  
`cd .\diabetes_active_proteins\`
- Install dependencies/packages/libraries/etc  
`pip install -r requirements.txt`

## Start the web application:
`streamlit run script.py`  
*(You can choose "test....csv" file from the `data` folder for testing purpose of this product).*
