# Example of how to use the trained model to predict the building structure of two buildings with the given area and story.

from pickle import load

# 'data/TrainedRF_noPOI.pkl' / 'data/TrainedRF.pkl'
TRAINED_RF = 'data/TrainedRF_noPOI.pkl' 

Area = [32000, 500]
Floor = [4, 10]
Footprint = [[(-80, -100), (80, -100), (80, 100), (-80, 100)], [(-12.5, -10), (12.5, -10), (12.5, 10), (-12.5, 10)]]

with open(TRAINED_RF, "rb") as f:

    clf = load(f)
    Y_test = clf.predict(Area, Floor, Footprint)

    print(Y_test)
