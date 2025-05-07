# Example of how to use the Class 'StructPred_RF'

import sys
from pathlib import Path
from pickle import dump

sys.path.insert(0, str(Path(__file__).parent.parent))
import BldStructPred
from BldStructPred.StructPred import StructPred_RF

DATA_DIR = Path(BldStructPred.__file__).parent / 'data'
DATA_FILE = DATA_DIR / '武汉建筑训练数据_POI_LJJ.csv'
N_VERT = 100 # number of vertices of the footprint
TRAINED_RF = DATA_DIR / 'TrainedRF.pkl'
TRAINED_RF_NOPOI = DATA_DIR / 'TrainedRF_noPOI.pkl'

for N_POI, out_file in zip([20, 0], [TRAINED_RF, TRAINED_RF_NOPOI]):
    clf = StructPred_RF(DATA_FILE, N_POI, N_VERT)
    clf.train()
    clf.evaluate()
    clf.plot_confusion_matrix()
    clf.plot_feature_importance()
    # Save the trained model
    with open(Path.cwd() / out_file, 'wb') as f:
        dump(clf, f)
