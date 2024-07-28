import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, Polygon

def _parse_poi(x):
    '''
    Parse the POI data
    Parameters:
        x: str, the POI data. e.g. "['汉北玺园;商务住宅;住宅区;住宅小区;汉口北大道附近;443.62614846789495', '联投·汉口郡;商务住宅;住宅区;住宅小区;汉口北大道特9号(胜海家园旁);521.3230907295101', ...]"
    returns:
        X: list, the parsed POI data. e.g. [443.62614846789495, '商务住宅', '住宅区', '住宅小区', 521.3230907295101, '商务住宅', '住宅区', '住宅小区', ...]
    '''
    x = eval(x)
    X = []
    for xx in x:
        site1 = xx.split(';')
        X.append(float(site1[5]))
        for i in range(1, 4):
            X.append(site1[i])
    return X

def _prepare_data(DATA_FILE, N_POI, N_vert):
    '''
    Prepare the training data
    Parameters:
        DATA_FILE: str, the path to the training data
        N_POI: int, the max number of POI sites
        N_vert: int, the max number of vertices of the footprint
    Returns:
        X_encoded: list, the features
        Y: list, the labels
        feature_names: list, the feature names
        EncCat: list, the label encoders for the categorical features
    '''

    DATA_FILE = Path.cwd() / DATA_FILE

    # Load the training data
    train_data: gpd.GeoDataFrame = gpd.read_file(DATA_FILE, encoding='utf-8-sig', GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
    train_data['POI'] = train_data['POI'].apply(_parse_poi)

    # Convert the CRS to UTM and calculate the area
    gdf_temp = train_data.set_crs(epsg=4326)
    utm_crs = gdf_temp.estimate_utm_crs()
    gdf_temp.to_crs(utm_crs, inplace=True)
    train_data['area'] = gdf_temp.area
    train_data['centroid'] = gdf_temp.centroid

    # Extract the footprint vertices (relative to the centroid)
    train_data['footprint'] = gdf_temp.geometry.apply(lambda x: [Point(p[0] - x.centroid.x, p[1] - x.centroid.y) for p in x.exterior.coords])
    # find the max number of footprint vertices
    N_max = max([len(x) for x in train_data['footprint']])
    if N_max < N_vert:
        for i in range(len(train_data)):
            N = len(train_data['footprint'][i])
            train_data.at[i, 'footprint'] = train_data['footprint'][i] + [train_data['footprint'][i][0]] * (N_vert - N)
    elif N_vert < N_max:
        Exception('The number of vertices of the footprint is larger than N_vert')

    # Expand the training data by duplicating them and dropping the POI data
    # dup_data = train_data.copy()
    # dup_data['POI'] = [[0,'','',''] * N_POI]*len(dup_data)
    # train_data = pd.concat([train_data, dup_data], ignore_index=True)

    # Extract the features and labels
    # features: [Area, floor, x coord of vert1, y coord of vert1, ..., Dist of POI1, Cat1 of POI1, Cat2 of POI1, Cat3 of POI1, Dist of POI2, Cat1 of POI2, Cat2 of POI2, Cat3 of POI2, ...]
    feature_names = ['Area', 'Floor']
    for i in range(N_vert):
        feature_names += [f'X_Vert_{i+1}', f'Y_Vert_{i+1}']
    for i in range(N_POI):
        feature_names += [f'Dist_{i+1}', f'Cat1_{i+1}', f'Cat2_{i+1}', f'Cat3_{i+1}']
    x_area = train_data['area'].astype(float).to_list()
    x_floor = train_data['floor'].astype(float).to_list()
    x_vert = [sum([[p.x, p.y] for p in x], []) for x in train_data['footprint']]
    x_poi = [ sum([[0,'','',''] for i in range(N_POI)], []) for _ in range(len(train_data))] # [Dist,Cat1,Cat2,Cat3] * N_POI
    for i in range(len(train_data)):
        end = min(len(train_data['POI'][i]), N_POI)
        x_poi[i][:end] = train_data['POI'][i][:end]
    X = [[c] + [b] + d + a for a, b, c, d in zip(x_poi, x_floor, x_area, x_vert)]
    Y = train_data['structure'].to_list()

    # create label encoder for each categorical feature
    EncCat = [LabelEncoder() for i in range(3)]
    for i in range(3):
        Cat1Data = []
        for a in x_poi:
            Cat1Data += a[-3+i::-4]
        EncCat[i].fit(Cat1Data)
    X_encoded = X
    for i in range(3):
        for irow in range(len(X_encoded)):
            X_encoded[irow][2+N_vert*2+1+i::4] = EncCat[i].transform(X_encoded[irow][2+N_vert*2+1+i::4])

    return X_encoded, Y, feature_names, EncCat

class StructClassify_RF:
    '''
    A class to train and evaluate a classifier for building structure classification
    Properties:
        data_file: str, the path to the training data
        n_poi: int, the max number of POI sites
        n_vert: int, the max number of vertices of the footprint
        X_encoded: list, the features
        Y: list, the labels
        feature_names: list, the feature names
        clf: Classifier, the trained classifier
    Private properties:
        _EncCat: list, the label encoders for the categorical features
    functions:
        train: Train a random forest classifier
        evaluate: Evaluate the random forest classifier
        plot_feature_importance: Plot the feature importance
        plot_confusion_matrix: Plot the confusion matrix
    '''

    def __init__(self, data_file, n_poi = 20, n_vert = 100):
        self.data_file = data_file
        self.n_poi = n_poi
        self.n_vert = n_vert

    def train(self):
        '''
        Train The models
        '''
        # Prepare the training data
        X_encoded, Y, feature_names, EncCat = _prepare_data(self.data_file, self.n_poi, self.n_vert)
        self.X_encoded = X_encoded
        self.Y = Y
        self.feature_names = feature_names
        self._EncCat = EncCat

        # Train a random forest classifier
        self.clf = RandomForestClassifier()
        self.clf.fit(X_encoded, Y)

    def predict(self, Area, Floor, Footprint, POI = None):
        '''
        Predict the building structural types of the given buildings
        Parameters:
            Area: list, the areas of the buildings. [n_samples]
            Floor: list, the number of floors of the buildings. [n_samples]
            Footprint: list, the footprints of the buildings. [n_samples, n_vertices]
                e.g. [[(x1, y1), (x2, y2), ...], ...]
                units are in meters
            POI: list, the POI data of the buildings. [n_samples, n_poi, 4]
                e.g. [[[Dist1, Cat1, Cat2, Cat3], ...], ...]
        Returns:
            Y_test: list, the predicted building structural types
        '''

        # make Footprint centered at the centroid
        Footprint = [ [[p[0] - Polygon(f).centroid.x, p[1] - Polygon(f).centroid.y] for p in f] for f in Footprint] # [[[x1, y1], [x2, y2], ...], ...]
        # make Footprint have the same number of vertices as n_vert
        N_max = max([len(x) for x in Footprint])
        if N_max <= self.n_vert:
            for i in range(len(Footprint)):
                N = len(Footprint[i])
                Footprint[i] = Footprint[i] + [Footprint[i][0]] * (self.n_vert - N)
        else:
            Exception('The number of vertices of the footprint is larger than N_vert')

        # Prepare the test data
        N = len(Area)
        X_test = [[Area[i], Floor[i]] + [xy for p in Footprint[i] for xy in p] + sum([[0, '', '', ''] for _ in range(self.n_poi)], []) for i in range(N)]
        if POI is not None:
            for i in range(N):
                end = min(len(POI[i]), self.n_poi)
                n_front = 2 + self.n_vert * 2
                X_test[i][n_front:n_front+end*4] = sum(POI[i][:end], [])
        else:
            if not self.n_poi == 0:
                Exception('The information of POI is not provided')
            # Encode the features
            X_test = self._encode_X(X_test)

        Y_test = self.clf.predict(X_test)

        return Y_test
    
    def _encode_X(self, X):
        '''
        Encode the features so that they can be used to predict the building structural types
        Parameters:
            X: list, the features of the buildings. 
                [n_samples, n_features]. 
                [[Area, floor, x coord of vert1, y coord of vert1, ..., Dist of POI1, Cat1 of POI1, Cat2 of POI1, Cat3 of POI1, Dist of POI2, Cat1 of POI2, Cat2 of POI2, Cat3 of POI2, ...], ...]. 
                e.g. [[5000, 3, ..., 221.3230907295101, '商务住宅', '住宅区', '住宅小区', 443.62614846789495, '商务住宅', '住宅区', '住宅小区', ...]]
                x coord of vert1, y coord of vert1, ... are the coordinates of the vertices of the footprint (centred at the centroid)
        '''
        X_encoded = X
        for i in range(3):
            for irow in range(len(X_encoded)):
                X_encoded[irow][2+self.n_vert*2+1+i::4] = self._EncCat[i].transform(X_encoded[irow][2+self.n_vert*2+1+i::4])
        return X_encoded

    def evaluate(self):
        '''
        Evaluate the random forest classifier
        '''
        # print accuracy
        clf = RandomForestClassifier()
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
        scores = cross_val_score(clf, self.X_encoded, self.Y, cv=cv)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        return scores.mean(), scores.std()

    def plot_feature_importance(self):
        # result = permutation_importance(self.clf, self.X_encoded, self.Y)
        # forest_importances = pd.Series(result.importances_mean, index=self.feature_names)
        forest_importances = pd.Series(self.clf.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.xticks(rotation=45)
        plt.xlim(-1, 20)
        plt.show()
    
    def plot_confusion_matrix(self):
        clf = RandomForestClassifier()
        y_pred = cross_val_predict(clf, self.X_encoded, self.Y)
        conf_mat = confusion_matrix(self.Y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=self.clf.classes_)
        disp.plot()
        plt.show()

    
