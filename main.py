from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.base._base import _clone_estimator
import tensorflow as tf
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from aeon.classification.sklearn import RotationForestClassifier, SklearnClassifierWrapper
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from aeon.utils.numba.general import z_normalise_series
from llm import GPTSelectShapelet
from data import TimeSeriesDataset
from aeon.classification.shapelet_based import ShapeletTransformClassifier
import random
import numpy as np
import torch
import sys
np.set_printoptions(suppress=True)
import os

RANDOM_STATE = 0

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    check_random_state(seed)
    RANDOM_STATE = seed



class TSC:
    def __init__(self, data, classification_model, 
                 name='ECG5000', info="The ECG5000 dataset consists of 5,000 two-second single-lead ECG records from both healthy and unhealthy individuals, including those with arrhythmias and myocardial ischemia. The data has been preprocessed to extract and interpolate each heartbeat to 140 time steps. It is labeled into five classes:  1 (Normal Heartbeat: regular and healthy heart activity),  2 (R-on-T Premature Ventricular Contraction: early ventricular contraction disrupting normal repolarization), 3 (Premature Ventricular Contraction: early ventricular contraction, often benign), 4 (Supraventricular Premature Beat: premature beat from the atria), and 5 (Unclassified Beat: beats requiring further investigation).", 
                 n_shapelet_samples=10000, max_shapelets=None, max_shapelet_length=None, batch_size=100, random_state=RANDOM_STATE, transform_limit_in_minutes=0, time_limit_in_minutes=0, n_jobs=1):

        self.name = name
        self.info = info
        self.X_train, self.y_train = data
        self.model = classification_model
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.batch_size = batch_size
        self.random_state = random_state
        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.n_jobs = n_jobs

        self._transformer = None
        self._estimator = None
        self._transform_limit_in_minutes = 0
        self._classifier_limit_in_minutes = 0
        self.X_t = None  


    
    def discover_shapelets(self):

        if self.time_limit_in_minutes > 0:
            
            third = self.time_limit_in_minutes / 3
            self._classifier_limit_in_minutes = third
            self._transform_limit_in_minutes = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            self._transform_limit_in_minutes = self.transform_limit_in_minutes

        self._transformer = RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            max_shapelet_length=self.max_shapelet_length,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            time_limit_in_minutes=self._transform_limit_in_minutes,
        )
        
        self.X_t = self._transformer.fit_transform(self.X_train, self.y_train)

    def select_shapelets(self):

        selector = GPTSelectShapelet(self.info)
        new_shapelets = []

        for shapelet in self._transformer.shapelets:
            raw_data = self.X_train[shapelet[4]][shapelet[3]]
            shapelet_unnormalise = self.X_train[shapelet[4]][shapelet[3]][shapelet[2] : shapelet[2]+shapelet[1]]

            temp_shapelet = shapelet[:6] + (shapelet_unnormalise, )

            score, reason = selector.load_data_prompt(temp_shapelet, raw_data)
            # if score >= 70:
            #     new_shapelets.append(shapelet)
            
            
            new_shapelets.append({
                'shapelet' : shapelet,
                'score' : score
            })
        
        return new_shapelets
        # self._transformer.shapelets = new_shapelets
        # self.X_t = self._transformer.fit_transform(self.X_train, self.y_train)

    def train_classifier(self):

        self._estimator = _clone_estimator(self.model, self.random_state)
        self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes
        self._estimator.fit(self.X_t, self.y_train)

    def evaluate_classifier(self, X_test, y_test):

        X_test_t = self._transformer.transform(X_test)
        y_pred = self._estimator.predict(X_test_t) 
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        return accuracy, f1

    def main(self, X_test, y_test):
        self.discover_shapelets()
        selection = self.select_shapelets()

        score_list = []
        for item in selection:
            score_list.append(item['score'])
        print(f"score for all:\n{score_list}\n")

        thesholds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        all_shapelet_num = len(self._transformer.shapelets)
        for theshold in thesholds:
            print(f"Experiment: threshold={theshold}")
            shapelets = []
            for item in selection:
                score = item['score']
                shapelet = item['shapelet']
                if score >= theshold:
                    shapelets.append(shapelet)
                

            print(f"From {all_shapelet_num} shapelets, select {len(shapelets)} whose score is above {theshold}")

            self._transformer.shapelets = shapelets
            self.X_t = self._transformer.fit_transform(self.X_train, self.y_train)

            self.train_classifier()
            accuracy, f1 = self.evaluate_classifier(X_test, y_test)
            
            print(f"Classification Accuracy: {accuracy} F1-score: {f1}")
            print('*' * 50)

def main(name = 'Handwriting', 
         info = "The Handwriting dataset captures the dynamic movements associated with writing individual letters and is widely used for tasks such as handwriting recognition and human-computer interaction. This dataset consists of 1000 cases, each represented by three-dimensional time series data with a length of 152 time steps, so dataset has three channels to describe the three accelerometer values. The dataset includes 26 classes, labeled numerically from 1.0 to 26.0, corresponding to the 26 letters of the English alphabet : the label 1.0 represents the letter 'A', 2.0 represents 'B', and so on, up to 26.0 for 'Z'. These labels allow researchers to classify and analyze the unique patterns associated with each letter's writing process. "):
    with tf.device('/GPU'):
        dataset = TimeSeriesDataset(dataset_name=name)
        X, y = dataset.get_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = SklearnClassifierWrapper(MLPClassifier(max_iter=1000, early_stopping=True))
        print(f"Start Experiment for {name}")
        model = TSC((X_train, y_train), classifier, name=name, info=info,n_shapelet_samples=75000 ,transform_limit_in_minutes=10, n_jobs=-1)
        model.main(X_test, y_test)

        
        model = ShapeletTransformClassifier(transform_limit_in_minutes=2, time_limit_in_minutes=4, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fone = f1_score(y_test, y_pred, average='macro')
        print(len(model._transformer.shapelets))
        print(f"Baseline Accuracy: {accuracy}  F1-score: {fone} (4,2)")

if __name__ == '__main__':
    set_seed(0)
    MultiDataset = {
        'Handwriting' : "The Handwriting dataset captures the dynamic movements associated with writing individual letters and is widely used for tasks such as handwriting recognition and human-computer interaction. This dataset consists of 1000 cases, each represented by three-dimensional time series data with a length of 152 time steps, so dataset has three channels to describe the three accelerometer values. The dataset includes 26 classes, labeled numerically from 1.0 to 26.0, corresponding to the 26 letters of the English alphabet : the label 1.0 represents the letter 'A', 2.0 represents 'B', and so on, up to 26.0 for 'Z'. These labels allow researchers to classify and analyze the unique patterns associated with each letter's writing process. ",
        'EthanolConcentration' : "The EthanolConcentration dataset captures the temporal changes in ethanol concentration levels over time and is widely used for tasks such as chemical process monitoring, quality control, and environmental sensing. This dataset consists of 1500 cases, each represented by a four-dimensional time series data with a length of 175 time steps, so the dataset has four channels to describe the four different sensor measurements. The dataset includes 4 classes, corresponding to different ethanol concentration levels: label 'e35', 'e38', 'e40' and 'e45' representing different ethanol concentrations from low to high. These labels allow researchers to classify and analyze the unique patterns associated with each ethanol concentration level, facilitating the development of accurate detection and classification models for ethanol-related applications.",
        'UWaveGestureLibrary' : "The UWaveGestureLibrary dataset captures the dynamic hand movements associated with performing specific gestures and is widely used for tasks such as gesture recognition, human-computer interaction, and wearable technology applications. This dataset consists of 440 cases, each represented by three-dimensional time series data with a length of 315 time steps. The dataset`s three channels describes the three accelerometer values (x, y, and z axes), capturing the motion in different spatial directions.The dataset includes 8 classes, labeled numerically from 1.0 to 8.0, corresponding to different gestures :  'X', 'CHECK', 'CIRCLE', 'VERTICAL UP', 'VERTICAL DOWN', 'HORIZONTAL LEFT', 'HORIZONTAL RIGHT' and 'DIAMOND'. These labels allow researchers to classify and analyze the unique patterns associated with each gesture's movement process, facilitating the development of accurate gesture recognition systems for various applications.",
        'Heartbeat' : "The Heartbeat dataset captures the physiological signals associated with heart activity and is widely used for tasks such as cardiac health monitoring, arrhythmia detection, and medical diagnostics. This dataset consists of 409 cases, each represented by 61-dimensional time series data with a length of 128 time steps, so the dataset has 61 channels to describe the electrocardiogram (ECG) signals. The dataset includes 2 classes, labeled 'normal' and 'abnormal'. These labels allow researchers to classify and analyze the unique patterns associated with each type of heartbeat, facilitating the development of accurate diagnostic tools for cardiovascular conditions.",
        'JapaneseVowels' : "The JapaneseVowels dataset captures the vocal patterns associated with the pronunciation of Japanese vowels and is widely used for tasks such as speech recognition. This dataset consists of 640 cases, each represented by 12-dimensional time series data with varying lengths, typically ranging from 7 to 29 time steps. The dataset has 12 channels, each representing a different cepstral coefficient derived from the vocal signal, capturing the unique acoustic features of the spoken vowels.The dataset includes 9 classes, labeled numerically from 1.0 to 9.0, corresponding to the nine different Japanese vowels pronounced by speakers: the label 1.0 represents the vowel 'a', 2.0 represents 'i', 3.0 represents 'u', 4.0 represents 'e', 5.0 represents 'o', and the remaining labels (6.0 to 9.0) correspond to variations or combinations of these vowels in different contexts. ",
        'SelfRegulationSCP1' : "The SelfRegulationSCP1 dataset records the cortical potential (EEG) signals of healthy subjects when moving a cursor on a computer screen. This dataset aims to study the changes in cortical slow potentials related to self-regulation. This dataset consists of 561 cases, each represented by 6-dimensional time series data with a length of 896 time steps. The dataset`s 6 channels corresponding to 6 EEG channels.The dataset is divided into two categories, labeled as 'negativity' (cursor down) and 'positivity' (cursor up).",
        'SelfRegulationSCP2' : "The SelfRegulationSCP2 dataset records the cortical potential (EEG) signals of amyotrophic lateral sclerosis (ALS) patients as they move the cursor on a computer screen. This dataset is used to study the changes in cortical slow potentials related to self-regulation, particularly in ALS patients.It has 380 cases, each represented by 7-dimensional time series data with a length of 1152 time steps. The 7 channels correspond to 7 EEG channels.The dataset is divided into two categories, labeled as 'negativity' (cursor down) and 'positivity' (cursor up).",
        'PEMS-SF' : "The PEMS-SF dataset is a widely used multivariate time series dataset for traffic flow analysis and prediction, which records vehicle occupancy information on different lanes of highways in the San Francisco Bay Area. This dataset consists of 400 cases, record every 10 minutes, from January 1, 2008 to March 30, 2009. The dataset contains 963 sensors as 963 channels, each of which records the vehicle occupancy rate within a day, ranging from 0 to 1.The dataset has 7 classes, each class corresponds to a day of the week (Monday to Sunday), with labels ranging from '1.0' to '7.0'.",
        'FaceDetection' : "The FaceDetection dataset is a multivariate time series dataset used for face detection, which includes features extracted from magnetoencephalography (MEG) records to distinguish between facial (Face) and non face (Scramble) signals.It has 9414 cases, divided into Face as label '1' and Scramble as label '0'. It has 144 feature dimensions, which may come from preprocessing and feature extraction of magnetoencephalography (MEG) signals and include time domain, frequency domain, or other statistical features used to describe the characteristics of the signal."
    }
    aim_dataset = ['PEMS-SF', 'EthanolConcentration']

    for name in aim_dataset:
        info = MultiDataset[name]
        log_file = f"{name}.log"
        with open(log_file, 'w') as log:
            original_stdout = sys.stdout
            sys.stdout = log
            main(name, info)
            sys.std = original_stdout
            
