#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script aims to train the U-Net and
    random forest model based on manual segmentation

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: Fabruary 2024
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model # type: ignore
from skimage.morphology import disk
from sklearn.model_selection import KFold
from keras import layers, Model, callbacks
from scipy.ndimage import maximum_filter as mf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage import io, feature, color, filters

from Utils import Time, Training, FeaturesExtraction

#%% Functions
# Define functions

def SetDirectories(Name:str) -> dict:

    """
    Return full path of the main project folders into a dictionary
    """

    CWD = str(Path.cwd())
    Start = CWD.find(Name)
    WD = Path(CWD[:Start], Name)
    Dirs = [D for D in WD.iterdir() if D.is_dir()]

    Directories = {}
    Directories['CWD'] = WD
    Directories['Data'] = [D for D in Dirs if 'Data' in D.name][0]
    Directories['Scripts'] = [D for D in Dirs if 'Scripts' in D.name][0]
    Directories['Results'] = [D for D in Dirs if 'Results' in D.name][0]

    return Directories
def ConvolutionBlock(Input:layers.Layer, nFilters:int):

    """
    Classical convolutional block used in simple U-net models
    """

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Input)
    Layer = layers.Activation("relu")(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Layer)
    Layer = layers.Activation("relu")(Layer)

    return Layer
def EncoderBlock(Input:layers.Layer, nFilters:int):

    """
    Classical encoder block used in simple U-net models
    """

    Layer = ConvolutionBlock(Input, nFilters)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def DecoderBlock(Input:layers.Layer, SkipFeatures:layers.Layer, nFilters:int):
    
    """
    Classical decoder block used in simple U-net models    
    """

    Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvolutionBlock(Layer, nFilters)
    return Layer
def BuildUnet(InputShape:tuple, nClasses:int, nFilters=[64, 128, 256, 512, 1024]) -> Model:

    """
    Builds simple U-net model for semantic segmentataion
    Model doesn't comprise dropout or batch normalization to keep
    architecture as simple as possible
    :param InputShape: tuple of 2 numbers defining the U-net input shape
    :param nClasses: integer defining the number of classes to segment
    :param nFilters: list of number of filters for each layer
    :return Unet: keras unet model
    """
    
    Input = layers.Input(InputShape)
    if len(nFilters) == 3:
        
        Block = []
        Block.append(EncoderBlock(Input, nFilters[0]))
        Block.append(EncoderBlock(Block[-1][1], nFilters[1]))

        Bridge = ConvolutionBlock(Block[-1][1], nFilters[2])
        D = DecoderBlock(Bridge, Block[-1][0], nFilters[1])
        D = DecoderBlock(D, Block[0][0], nFilters[0])

    else:
        Block = []
        Block.append(EncoderBlock(Input, nFilters[0]))
        for i, nFilter in enumerate(nFilters[1:-1]):
            Block.append(EncoderBlock(Block[i][1], nFilter))

        Bridge = ConvolutionBlock(Block[-1][1], nFilters[-1])
        D = DecoderBlock(Bridge, Block[-1][0], nFilters[-2])

        for i, nFilter in enumerate(nFilters[-3::-1]):
            D = DecoderBlock(D, Block[-i+2][0], nFilter)

    # If binary classification, uses sigmoid activation function
    if nClasses == 2:
      Activation = 'sigmoid'
    else:
      Activation = 'softmax'

    Outputs = layers.Conv2D(nClasses, 1, padding="same", activation=Activation)(D)
    Unet = Model(Input, Outputs, name='U-Net')
    return Unet

#%% Main
# Main part

def Main():

    # List training data
    Dirs = SetDirectories('FEXHIP-Histology')
    ResultsDir = Dirs['Results'] / '01_Training'
    DataPath = ResultsDir / 'Preprocessing'
    Images, Labels = Training.GetData(DataPath)

    # Get sample weights
    sWeights, YData = Training.SetWeights(Labels)

    # Define class weights
    Counts = np.sum(YData, axis=(0,1,2))
    cWeights = 1 / (Counts / sum(Counts) * len(Counts))

    # Split into train and test data
    XTrain, XTest, YTrain, YTest = train_test_split(Images/255, YData, random_state=42)
    WTrain, WTest = train_test_split(sWeights, random_state=42)

    # Build UNet
    Unet = BuildUnet(XTrain.shape[1:], YTrain.shape[-1], nFilters=[8, 16, 32])
    Unet.compile(optimizer='adam',
                 loss='binary_focal_crossentropy',
                 metrics=['accuracy'],
                 loss_weights=[cWeights],
                 weighted_metrics=['accuracy']
                 )
    print(Unet.summary())
    EarlyStop = callbacks.EarlyStopping(monitor='accuracy', patience=100)
    ModelName = str(ResultsDir / 'UNet_reduced.hdf5')
    CheckPoint = callbacks.ModelCheckpoint(ModelName, monitor='accuracy',
                                           mode='max', save_best_only=True)
    History = Unet.fit(XTrain,YTrain, validation_data=(XTest,YTest,WTest),
                       verbose=2, epochs=250, workers=8,
                       batch_size=32, steps_per_epoch=20,
                       callbacks=[EarlyStop, CheckPoint],
                       sample_weight=WTrain)
    HistoryDf = pd.DataFrame(History.history)
    HistoryDf.to_csv(ResultsDir / 'History.csv', index=False)

    # Plot history
    HistoryDf = pd.read_csv(ResultsDir / 'History.csv')
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(HistoryDf['weighted_accuracy'], color=(1,0,0), label='Training data')
    Axis.plot(HistoryDf['val_weighted_accuracy'], color=(0,0,1), label='Validation data')
    plt.legend()
    plt.show(Figure)

    # Load best model
    Unet = load_model(ResultsDir / 'UNet.hdf5')

    # Select models outputs for features extractor
    Outputs = [L.output for L in Unet.layers if 'conv' in L.name]
    FeaturesExtractor = Model(Unet.input, Outputs)

    # Features extraction
    SubFeatures = []
    SubLabels = []
    SubWeights = []
    Time.Process(1,'Features extraction')
    for i, xTrain in enumerate(XTrain):
        XF = FeaturesExtraction(FeaturesExtractor, xTrain)
        Features = XF.reshape(-1, XF.shape[-1])

        # Balance and subsample data for faster fit
        aLabels = np.argmax(YTrain[i], axis=-1)
        Values, Counts = np.unique(aLabels, return_counts=True)
        Indices = pd.DataFrame(aLabels.ravel()).groupby(0).sample(min(Counts)).index

        # Subsample
        SubFeatures.append(Features[Indices])
        SubLabels.append(aLabels.ravel()[Indices])
        SubWeights.append(WTrain[i].ravel()[Indices])

        Time.Update((i+1) / len(XTrain))

    # Concatenate lists
    SubFeatures = np.concatenate(SubFeatures, axis=0)
    SubLabels = np.concatenate(SubLabels, axis=0)
    SubWeights = np.concatenate(SubWeights, axis=0)

    Time.Process(0)

    # Instanciate random forest classifier
    RFc = RandomForestClassifier(n_estimators=100,
                                 oob_score=True,
                                 max_depth=10,
                                 n_jobs=-1,
                                 verbose=2,
                                 class_weight='balanced')

    # Fit random forest classifier with all training data and save it
    RFc.fit(SubFeatures, SubLabels.ravel()+1, sample_weight=SubWeights)
    joblib.dump(RFc, str(ResultsDir / 'RandomForest.joblib'))
    RFc = joblib.load(str(ResultsDir / 'RandomForest.joblib'))

    # Look at random test image
    Random = np.random.randint(XTest.shape[0])
    Test = XTest[Random]
    XF = FeaturesExtraction(FeaturesExtractor, Test)
    Features = XF.reshape(-1, XF.shape[-1])

    Prediction = RFc.predict(Features)
    Prediction = np.reshape(Prediction,Test.shape[:-1])
    uPrediction = Unet.predict(np.expand_dims(Test,0))[0]

    Figure, Axis = plt.subplots(2,2, figsize=(8,8))
    Axis[0,0].imshow(Test)
    Axis[0,1].imshow(YTest[Random][:,:,1:]*255)
    Axis[1,0].imshow(uPrediction[:,:,1:])
    Axis[1,1].imshow(Prediction)
    for i in range(2):
        for j in range(2):
            Axis[i,j].axis('off')
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    plt.show(Figure)

    return

#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
# %%
