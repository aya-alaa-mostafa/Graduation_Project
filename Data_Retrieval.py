import numpy as np
import pandas as pd
from featurize import generate_descriptors, generate_fingerprints, extract_smiles, extract_smile_structures
from pubchem import get_assay_results, get_smile
def create_labeled_dataset():
    assay_data = get_assay_results(aid='1851', tids={60, 68})
    assay_data = pd.DataFrame(assay_data, columns=['sid', 'score', 'curve_class'], dtype='object')
    smiles = get_smile(sids=assay_data.iloc[:, 0].astype(int))
    smiles = pd.DataFrame(smiles, columns=['cid', 'smile'])
    assay_data = pd.concat((smiles.smile, assay_data.iloc[:, [1, 2]]), axis=1).dropna()
    inhibitor = assay_data.loc[(assay_data.score >= 40) & assay_data.curve_class.isin({-1.1, -1.2, -2.1}), ['smile']]
    inhibitor['label'] = 'inhibitor'
    noninhibitor = assay_data.loc[(assay_data.score == 0) & (assay_data.curve_class == 4), ['smile']]
    noninhibitor['label'] = 'noninhibitor'
    return pd.concat((inhibitor, noninhibitor), axis=0).drop_duplicates('smile').reset_index(drop=True)
labeled_data = create_labeled_dataset()
mordred_features = generate_descriptors(labeled_data.smile.to_list())
fingerprints = generate_fingerprints(labeled_data.smile)
labeled_data = pd.concat([labeled_data, mordred_features, fingerprints], axis=1)
labeled_data.to_csv('data/cyp3a4_labeled_data.csv', index=False)
smile_features = extract_smiles(labeled_data.smile, max_length=250)
np.save('data/cyp3a4_smile_features', smile_features)
smile_structure = extract_smile_structures(labeled_data.smile, resolution=100, scale=(-15, 15))
np.save('data/cyp3a4_smile_structure', smile_structure)