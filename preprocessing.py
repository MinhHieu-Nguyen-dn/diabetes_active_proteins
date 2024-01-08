import math
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split


def get_dataset(uniprot_id, data_dir='data'):
    """
    Connect to ChEMBL to find, get, and save target and compound activity data
    to the input data directory.

    :param uniprot_id: UniProt ID of the target of interest from UniProt website.
    :param data_dir: Path to data directory.
    :return:
    """
    # Create resource objects for API access.
    targets_api = new_client.target
    compounds_api = new_client.molecule
    bioactivities_api = new_client.activity

    # Get target information from ChEMBL but restrict it to specified values only
    targets = targets_api.get(target_components__accession=uniprot_id).only(
        "target_chembl_id", "organism", "pref_name", "target_type"
    )
    # Download target data from ChEMBL
    targets = pd.DataFrame.from_records(targets)
    # Select target
    target = targets.iloc[0]
    # Save selected ChEMBL ID
    chembl_id = target.target_chembl_id

    # Fetch bioactivity data for the target from ChEMBL
    bioactivities = bioactivities_api.filter(
        target_chembl_id=chembl_id, type="IC50", relation="=", assay_type="B"
    ).only(
        "activity_id",
        "assay_chembl_id",
        "assay_description",
        "assay_type",
        "molecule_chembl_id",
        "type",
        "standard_units",
        "relation",
        "standard_value",
        "target_chembl_id",
        "target_organism",
    )

    # Download bioactivity data from ChEMBL (QuerySet) in the form of a pandas DataFrame.
    bioactivities_df = pd.DataFrame.from_dict(bioactivities)
    # Drop "units" and "value" column to use standard values and unit only.
    bioactivities_df.drop(["units", "value"], axis=1, inplace=True)

    # Preprocessing
    bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
    # Delete entries with missing values
    bioactivities_df.dropna(axis=0, how="any", inplace=True)
    # Keep entries with unit nM only
    bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
    # Delete duplicate molecules
    bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    # Reset dataframe index
    bioactivities_df.reset_index(drop=True, inplace=True)
    # Rename columns
    bioactivities_df.rename(
        columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True
    )

    # Get compound data
    # Fetch compound data from ChEMBL
    compounds_provider = compounds_api.filter(
        molecule_chembl_id__in=list(bioactivities_df["molecule_chembl_id"])
    ).only("molecule_chembl_id", "molecule_structures")

    # Download compound data from ChEMBL: export the QuerySet object into a pandas DataFrame
    compounds = list(tqdm(compounds_provider))
    compounds_df = pd.DataFrame.from_records(
        compounds,
    )

    # Preprocessing
    # Remove missing entries
    compounds_df.dropna(axis=0, how="any", inplace=True)
    # Delete duplicates
    compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
    # Keep only the canonical SMILES structure
    canonical_smiles = []

    for i, compounds in compounds_df.iterrows():
        try:
            canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
        except KeyError:
            canonical_smiles.append(None)

    compounds_df["smiles"] = canonical_smiles
    compounds_df.drop("molecule_structures", axis=1, inplace=True)
    compounds_df.dropna(axis=0, how="any", inplace=True)

    # Combine: bioactivity-compound data
    print(f"Bioactivities filtered: {bioactivities_df.shape[0]}")
    print(f"Compounds filtered: {compounds_df.shape[0]}")

    # Merge DataFrames
    output_df = pd.merge(
        bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
        compounds_df,
        on="molecule_chembl_id",
    )

    # Reset row indices
    output_df.reset_index(drop=True, inplace=True)

    print(f"Dataset with {output_df.shape[0]} entries.")

    # Save file
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    save_path = os.path.join(data_dir, '{}.csv'.format(uniprot_id))
    output_df.to_csv(save_path)
    print('Get and save data: done with {}'.format(uniprot_id))


def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value


def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.

    """

    # convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    if method == "morgan3":
        fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        return np.array(fpg.GetFingerprint(mol))
    else:
        # NBVAL_CHECK_OUTPUT
        print(f"Warning: Wrong method specified: {method}. Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))


def prepare_data_for_model(uniprot_id, data_dir='data'):
    """

    :param uniprot_id:
    :param data_dir:
    :return:
    """
    file_path = os.path.join('data', '{}.csv'.format(uniprot_id))
    df = pd.read_csv(file_path, index_col=0)
    # Add new column for fingerprints
    df["fp"] = df["smiles"].apply(smiles_to_fp)
    # Apply conversion IC50 to pIC50 to each row of the compounds DataFrame
    df["pIC50"] = df.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)
    # Create active column
    df["active"] = df["pIC50"].apply(lambda x: 1 if x > 6 else 0)

    # Save and return result
    print("Added fp, pIC50, and active columns to {}".format(file_path))
    new_file_path = os.path.join('data', 'processed_{}.csv'.format(uniprot_id))
    print("Saved as {}".format(new_file_path))
    df.to_csv(new_file_path)
    return df


def string_to_np_array(s: str):
    s = s.strip('[]')
    arr = np.fromstring(s, sep=' ')

    return arr


def df_to_data_split(df, random_seed=1):
    fingerprint_to_model = df.fp.tolist()
    label_to_model = df.active.tolist()

    (
        static_train_x,
        static_test_x,
        static_train_y,
        static_test_y,
    ) = train_test_split(fingerprint_to_model, label_to_model, test_size=0.2, random_state=random_seed)

    splits = {
        'x_train': static_train_x,
        'x_test': static_test_x,
        'y_train': static_train_y,
        'y_test': static_test_y
    }

    return splits

# get_dataset(uniprot_id='P15121')
# get_dataset(uniprot_id='P31639')
#
# prepare_data_for_model(uniprot_id='P15121')
# prepare_data_for_model(uniprot_id='P31639')
