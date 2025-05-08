# Example of how to use the Class 'StructPred_RF'

import sys
from pathlib import Path
from joblib import dump
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import BldStructPred
from BldStructPred.StructPred import StructPred_RF

DATA_DIR = Path(BldStructPred.__file__).parent / 'data'
DATA_FILE = DATA_DIR / '武汉建筑训练数据_POI_LJJ.csv'
N_VERT = 100 # number of vertices of the footprint

np_version = np.__version__
if np_version.startswith('1.26'):
    TRAINED_RF = DATA_DIR / 'TrainedRF_numpy_v_1_26.joblib'
    TRAINED_RF_NOPOI = DATA_DIR / 'TrainedRF_noPOI_numpy_v_1_26.joblib'
else:
    TRAINED_RF = DATA_DIR / 'TrainedRF.joblib'
    TRAINED_RF_NOPOI = DATA_DIR / 'TrainedRF_noPOI.joblib'

for N_POI, out_file in zip([20, 0], [TRAINED_RF, TRAINED_RF_NOPOI]):
    clf = StructPred_RF(DATA_FILE, N_POI, N_VERT)
    clf.train()
    clf.evaluate()
    clf.plot_confusion_matrix()
    clf.plot_feature_importance()
    # Save the trained model
    dump(clf, out_file)
    print(f"Trained model saved to {out_file}")
