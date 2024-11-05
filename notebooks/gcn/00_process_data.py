import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import Descriptors

# The following functions are from the work of Feinstein et al.
def count_cf_bonds(mol):
    abstract_cf = Chem.MolFromSmarts('C~F')
    cf_bonds = mol.GetSubstructMatches(abstract_cf)
    return len(cf_bonds)

# Turn to EPA categories
def convert_to_mgkg(neglogld50s, smiles):
    mgkg_values = []
    for neglogld50, smile in zip(neglogld50s, smiles):
        molwt = Descriptors.MolWt(Chem.MolFromSmiles(smile))
        mgkg = (10**(-1*neglogld50)) * 1000 * molwt
        mgkg_values.append(mgkg)
    return mgkg_values

# Function to convert mg/kg values to EPA categories
def convert_to_epa(neglog_values, smiles):
    mgkg_values = convert_to_mgkg(neglog_values, smiles)
    epa_categories = pd.cut(mgkg_values, labels=[0,1,2,3], bins=[-np.inf, 50, 500, 5000, np.inf])
    return epa_categories

def safe_smiles(smiles_series, remove_stereochemistry = True):
    """
    Converts a series of SMILES strings into canonical SMILES after validating 
    the conversion from SMILES to molecule and back to SMILES.
    
    Parameters:
    smiles_series (pd.Series): A pandas Series containing SMILES strings.
    
    Returns:
    pd.Series: A pandas Series containing canonical SMILES strings, 
               or None for invalid SMILES.
    """
    def safe_smiles_to_smiles(smiles, idx):
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Convert molecule back to canonical SMILES but
                # first get rid off stereochemistry
                if remove_stereochemistry:
                    return Chem.MolToSmiles(mol, isomericSmiles=False)
                else:
                    return Chem.MolToSmiles(mol)
            else:
                raise ValueError("Invalid molecule")
        except Exception as e:
            # Handle any errors and log the index and SMILES string
            print(f"Error at index {idx}: processing SMILES {smiles}, Error: {e}")
            return None
    
    return smiles_series.apply(lambda smiles: safe_smiles_to_smiles(smiles, smiles_series.index[smiles_series == smiles][0]))

if __name__ == "__main__":
    # Read the data with smiles and LD50
    ldtoxdb = pd.read_csv('../../data/ldtoxdb-mordred.csv').dropna(axis=1)
    # SMILES CANONIZATION
    ldtoxdb['smiles'] = safe_smiles(ldtoxdb.SMI)
    ldtoxdb = ldtoxdb.dropna(subset=['smiles'])

    # Get other info
    ldtoxdb['rd_mol'] = ldtoxdb.smiles.apply(Chem.MolFromSmiles)
    ldtoxdb['mol_wt'] = ldtoxdb.rd_mol.apply(Chem.Descriptors.MolWt)

    # Find PFAS like
    ldtoxdb['n_cf_bonds'] = ldtoxdb.rd_mol.apply(count_cf_bonds)
    ldtoxdb['is_pfas_like'] = ldtoxdb['n_cf_bonds'] >= 2

    ldtoxdb = ldtoxdb.drop_duplicates(subset='smiles', keep='first')

    # Read the PFAS dataset and convert smiles
    pfas8k = pd.read_csv('../../data/pfas8k-mordred.csv')
    pfas8k['canon_smi'] = safe_smiles(pfas8k.SMILES)
    pfas8k = pfas8k.dropna(subset=['canon_smi'])
    pfas8k = pfas8k.drop_duplicates(subset='canon_smi', keep='first')

    # Classify the LDTOXDB
    ldtoxdb['is_pfas'] = ldtoxdb.smiles.isin(pfas8k.canon_smi)
    # Add EPA classes
    ldtoxdb['EPA'] = convert_to_epa(ldtoxdb['NeglogLD50'], smiles=ldtoxdb['smiles'])

    ldtoxdb.columns = ldtoxdb.columns.str.lower()

    ldtoxdb.to_csv('../../data/full_dataset.csv', index=False)

    # Separate PFAS and PFAS-like from data
    pfas_test = ldtoxdb[(ldtoxdb['is_pfas']) | (ldtoxdb['is_pfas_like'])]

    # The rest of the DataFrame where both columns are False
    training = ldtoxdb[~((ldtoxdb['is_pfas']) | (ldtoxdb['is_pfas_like']))]

    training.to_csv('../../data/training_dataset.csv', index=False)
    pfas_test.to_csv('../../data/test_pfas_dataset.csv', index=False)
