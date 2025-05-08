# Example of how to use the trained model to predict the building structure of two buildings with the given area and story.

from joblib import load
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
import BldStructPred
from packaging import version

data_dir = Path(BldStructPred.__file__).parent / 'data'

np_version_obj = version.parse(np.__version__)
if np_version_obj < version.parse('1.27.0'):
    TRAINED_RF = data_dir / 'TrainedRF_numpy_v_1_26.joblib'
else:
    TRAINED_RF = data_dir / 'TrainedRF.joblib'

Area = [32000, 500]
Floor = [4, 10]
Footprint = [[(-80, -100), (80, -100), (80, 100), (-80, 100)], [(-12.5, -10), (12.5, -10), (12.5, 10), (-12.5, 10)]]
# POI: list, the POI data of the buildings. [n_samples, n_poi, 4] e.g. [[[Dist1, Cat1, Cat2, Cat3], ...], ...]
POI = [[[443.62614846789495, '商务住宅','住宅区','住宅小区']], [[294.6821277194221,'商务住宅','住宅区','住宅小区']]]


clf = load(TRAINED_RF)
Y_test = clf.predict(Area, Floor, Footprint, POI)
print(Y_test)
