# Example of how to use the trained model to predict the building structure of two buildings with the given area and story.

from pickle import load
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import BldStructPred
from BldStructPred.StructPred import StructPred_RF

# 'data/TrainedRF_noPOI.pkl' / 'data/TrainedRF.pkl'
TRAINED_RF = Path(BldStructPred.__file__).parent / 'data/TrainedRF.pkl' 

Area = [32000, 500]
Floor = [4, 10]
Footprint = [[(-80, -100), (80, -100), (80, 100), (-80, 100)], [(-12.5, -10), (12.5, -10), (12.5, 10), (-12.5, 10)]]
# POI: list, the POI data of the buildings. [n_samples, n_poi, 4] e.g. [[[Dist1, Cat1, Cat2, Cat3], ...], ...]
POI = [[[443.62614846789495, '商务住宅','住宅区','住宅小区']], [[294.6821277194221,'商务住宅','住宅区','住宅小区']]]

with open(TRAINED_RF, "rb") as f:

    clf = load(f)
    Y_test = clf.predict(Area, Floor, Footprint, POI)

    print(Y_test)
