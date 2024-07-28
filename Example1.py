# Example of how to use the Class 'StructClassify_RF'

from BldStructClassify.StructClassify import StructClassify_RF
from pathlib import Path
from pickle import dump

DATA_FILE = 'data/武汉建筑训练数据_POI.csv'
N_VERT = 100 # number of vertices of the footprint
TRAINED_RF = 'data/TrainedRF.pkl'
TRAINED_RF_noPOI = 'data/TrainedRF_noPOI.pkl'

for N_POI, out_file in zip([20, 0], [TRAINED_RF, TRAINED_RF_noPOI]):
    clf = StructClassify_RF(DATA_FILE, N_POI, N_VERT)
    clf.train()
    clf.evaluate()
    clf.plot_confusion_matrix()
    clf.plot_feature_importance()
    # Save the trained model
    with open(Path.cwd() / out_file, 'wb') as f:
        dump(clf, f)
