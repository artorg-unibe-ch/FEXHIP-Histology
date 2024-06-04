#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script aims to test the module instalation

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn import metrics
from keras import utils, Model
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.morphology import disk
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from skimage import io, feature, morphology, measure

from Utils import Time, FeaturesExtraction, StainNA, Reference, CVAT, Unet_Probabilities

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
def ShowKFolds(Results:pd.DataFrame, Sites:list, FileName:str) -> None:

    """
    Plot (barplot) results of the K-folds cross validation
    """

    NSplits = len(Results)

    for Metric in ['Precision', 'Recall']:
        
        Figure, Axis = plt.subplots(2,2, sharex=True, sharey=True, dpi=200)
        for i in range(2):
            for j in range(2):
                # Plot bars
                Axis[i,j].bar(np.arange(NSplits)+1,
                            Results[Metric][Sites[i*2+j]],
                            edgecolor=(0,0,1), color=(1, 1, 1, 0))

                # Plot mean value
                Mean = Results[Metric][Sites[i*2+j]].mean()
                Axis[i,j].plot(np.arange(NSplits)+1,
                            np.repeat(Mean, NSplits),
                            color=(1,0,0))
                
                # Plot standard deviation range
                Std = Results[Metric][Sites[i*2+j]].std()
                Axis[i,j].fill_between(np.arange(NSplits)+1,
                                    np.repeat(Mean - Std, NSplits),
                                    np.repeat(Mean + Std, NSplits),
                                    color=(0,0,0,0.2))
                
                # Set axis label and title
                Axis[i,j].set_title(Sites[i*2+j])

                Axis[1,j].set_xlabel('Folds')
                Axis[1,j].set_xticks(np.arange(NSplits)+1)
                Axis[i,0].set_ylabel(Metric)
        plt.savefig(FileName + '_' + Metric + '.png', dpi=200)
        plt.show()

    return
def ClassesDistribution(YData:np.array) -> None:

    """
    Plot (barplot) distribution of the classes in YData
    """

    Values, Counts = np.unique(YData, return_counts=True)

    Figure, Axis = plt.subplots(1,1, figsize=(4,4), dpi = 200)
    Axis.bar(Values, Counts, edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.set_xticks(Values,[ 'Interstitial\nTissue', 'Osteocyte', 'Haversian\nCanal','Cement\nLine'])
    Axis.set_ylabel('Number of pixels (-)')
    Axis.set_title(f'Total {sum(Counts)} pixels')
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.show(Figure)

    return
# def GetNeighbours(Array:np.array) -> (np.array, np.array):
#     """
#     Function used to get values of the neighbourhood pixels (based on numpy.roll)
#     :param Array: numpy array
#     :return: Neighbourhood pixels values
#     """

#     # Define a map for the neighbour index computation
#     Map = np.array([[-1, 0], [ 1, 0],
#                     [ 0,-1], [ 0, 1],
#                     [-1,-1], [ 1, 1],
#                     [-1, 1], [ 1,-1]])
    
#     Neighbourhood = np.zeros(Array.shape + (len(Map),))
#     PadArray = np.pad(Array, ((1, 1), (1, 1)))

#     for i, Shift in enumerate(Map):
#         Neighbourhood[:,:,i] = np.roll(PadArray, Shift, axis=(0,1))[1:-1,1:-1]

#     return Neighbourhood, Map
# def RemoveEndBranches(Skeleton:np.array) -> np.array:

#     """
#     Remove iterativerly endbranches of a skeleton image
#     """

#     N = GetNeighbours(Skeleton)[0]
#     EndPoints = N.sum(axis=-1) * Skeleton == 1
#     while np.sum(EndPoints):
#         Skeleton[EndPoints] = False
#         N = GetNeighbours(Skeleton)[0]
#         EndPoints = N.sum(axis=-1) * Skeleton == 1

#     return Skeleton

class Visualize():

    """
    Bundle of functions used to look into what sort of
    visual patterns image classification models learn
    Based on Keras tutorial:
    https://keras.io/examples/vision/visualizing_what_convnets_learn/
    """

    def __init__(self,
                 Features_Extractor:Model,
                 Layer, Iterations=30,
                 LearningRate=10.0,
                 Show=True
                 ) -> None:
        
        self.Features_Extractor = Features_Extractor
        self.Layer = Layer
        self.Iterations = Iterations
        self.LearningRate = LearningRate
        self.Show = Show

        return

    def InitializeImage(self):

        Shape = self.Features_Extractor.input_shape
        Width, Height = Shape[1:3]

        # Start from a gray image with some random noise
        Random = tf.random.uniform((1, Width, Height, 3))

        # Scale random inputs to [-0.375, +0.625]
        return (Random - 0.5) * 0.25 + 0.5

    def ComputeLoss(self, Image:np.array, FIndex:int) -> float:
        """
        The "loss" we will maximize is simply the mean of the activation
        of a specific filter in our target layer. To avoid border effects,
        we exclude border pixels.
        """

        Activation = self.Features_Extractor(Image)
        LActivation = Activation[self.Layer]

        # We avoid border artifacts by only involving non-border pixels in the loss.
        FActivation = LActivation[:, 2:-2, 2:-2, FIndex]
        return tf.reduce_mean(FActivation)
    
    @tf.function
    def GA_Step(self, Image:np.array, FIndex:int):

        """
        The gradient ascent function simply computes the gradients of the loss
        with regard to the input image, and update the image so as to move it
        towards a state that will activate the target filter more strongly
        """

        with tf.GradientTape() as Tape:
            Tape.watch(Image)
            Loss = self.ComputeLoss(Image, FIndex)

        # Compute gradients
        Gradients = Tape.gradient(Loss, Image)

        # Normalize gradients
        Gradients = tf.math.l2_normalize(Gradients)

        # Update image
        Image += self.LearningRate * Gradients

        return Loss, Image
    
    def DeprocessImage(self, Image):
        
        # Normalize array: center on 0., ensure variance is 0.15
        Image -= Image.mean()
        Image /= Image.std() + 1e-5
        Image *= 0.15

        # Center crop
        Image = Image[25:-25, 25:-25, :]

        # Clip to [0, 1]
        Image += 0.5
        Image = np.clip(Image, 0, 1)

        # Convert to RGB array
        Image *= 255
        Image = np.clip(Image, 0, 255).astype("uint8")
        return Image
    
    def Filter(self, FIndex:int):

        # Iteration gradient ascent step
        Image = self.InitializeImage()
        for Iteration in range(self.Iterations):
            Loss, Image = self.GA_Step(Image, FIndex)

        # Decode the resulting input image
        Image = self.DeprocessImage(Image[0].numpy())

        # Show the images
        if self.Show:
            Figure, Axis = plt.subplots(1,1)
            Axis.imshow(Image)
            Axis.axis('off')
            plt.show()

        return Loss, Image

#%% Main
# Main part

def Main():

    # Set directories
    Dirs = SetDirectories('FEXHIP-Histology')
    TrainingDir = Dirs['Data'] / 'Training'
    ModelDir = Dirs['Results'] / '01_Training'
    ResultsDir = ModelDir / 'Assessment'
    os.makedirs(ResultsDir, exist_ok=True)

    # List training data
    Names, Images, Labels = CVAT.GetData(TrainingDir)

    # Get common ROI
    CommonName, CommonROI, SegRatio, Indices = CVAT.CommonROI()

    # Collect individual segments
    OneHot = CVAT.OneHot(Labels)

    # Store operator score for each segment
    sWeights = CVAT.SampleWeights(OneHot, Indices, SegRatio)

    # Get data, one-hot encoded labels, sample and classes weights
    YData = np.argmax(OneHot, axis=-1) + 1

    # Keep only 1 occurence of common ROI
    ROIs = Images[~Indices]
    Masks = np.expand_dims(sWeights[~Indices],-1) * OneHot[~Indices]
    Truth = YData[~Indices]

    ROIs = np.concatenate([np.expand_dims(CommonROI, 0), ROIs])
    sWeights = np.concatenate([np.expand_dims(SegRatio, 0), Masks])
    YData = np.concatenate([np.expand_dims(YData[Indices][0], 0), Truth])

    # Define data
    Stain = [StainNA(R, Reference.Mean, Reference.Std) for R in ROIs]
    XData = np.array(Stain)

    # Load u-net model
    Unet = load_model(ModelDir / 'UNet.hdf5')

    # Select models outputs for features extractor
    Outputs = [L.output for L in Unet.layers if 'conv' in L.name]
    FeaturesExtractor = Model(Unet.input, Outputs)

    # Features extraction
    SubFeatures = []
    SubLabels = []
    SubWeights = []
    Time.Process(1,'Features extraction')
    for i, xData in enumerate(XData):
        FE = FeaturesExtraction(FeaturesExtractor, xData)

        # Balance and subsample data for faster fit
        Values, Counts = np.unique(YData[i], return_counts=True)
        Indices = pd.DataFrame(YData[i].ravel()).groupby(0).sample(min(Counts)).index

        # Reshape array and subsample
        Features = FE.reshape(-1, FE.shape[-1])
        SubFeatures.append(Features[Indices])
        SubLabels.append(YData[i].ravel()[Indices])
        SubWeights.append(sWeights[i].ravel()[Indices])

        Time.Update((i+1) / len(XData))

    # Concatenate lists
    SubFeatures = np.concatenate(SubFeatures, axis=0)
    SubLabels = np.concatenate(SubLabels, axis=0)
    SubWeights = np.concatenate(SubWeights, axis=0)

    Time.Process(0)

    # Instanciate random forest classifier
    RFc = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                n_jobs=-1,
                                verbose=0,
                                max_depth=10,
                                class_weight='balanced')

    # Use KFold cross validation
    NSplits = 7
    CV = KFold(n_splits=NSplits, shuffle=True, random_state=42)
    FoldsScores = []
    Time.Process(1, 'KFold validation')
    for i, (Train, Test) in enumerate(CV.split(SubFeatures, SubLabels)):

        # Get train and test data
        XTrain = SubFeatures[Train]
        YTrain = SubLabels[Train]
        WTrain = SubWeights[Train]
        XTest = SubFeatures[Test]
        YTest = SubLabels[Test]

        # Fit random forest
        RFc.fit(XTrain, YTrain, sample_weight=WTrain)

        # Perform prediction
        Pred = RFc.predict(XTest)
        
        # Compute scores
        RF_Precision = metrics.precision_score(Pred, YTest, average=None)
        RF_Recall = metrics.recall_score(Pred, YTest, average=None)
        Precision = [P for P in RF_Precision]
        Recall = [R for R in RF_Recall]
        FoldsScores.append([Precision, Recall])

        # Update time
        Time.Update(i / NSplits)
    Time.Process(0)

    # Save scores and plot
    Sites = ['Interstitial Tissue', 'Osteocytes', 'Haversian Canals', 'Cement Lines']
    Cols = pd.MultiIndex.from_product([['Precision','Recall'], Sites])
    FoldsScores = np.array(FoldsScores).reshape(NSplits, len(Cols))
    FoldsScores = pd.DataFrame(FoldsScores, columns=Cols)
    FoldsScores.to_csv(str(ResultsDir / 'FoldsScores.csv'), index=False)
    ShowKFolds(FoldsScores, Sites, str(ResultsDir / 'KFolds'))

    # Load model fitted with all data and segment original images
    RFc = joblib.load(str(ModelDir / 'RandomForest.joblib'))
    RFc.verbose = 0
    Prediction = []
    Time.Process(1,'Predict Segmentation')
    for i, xData in enumerate(XData):
        FE = FeaturesExtraction(FeaturesExtractor, xData)
        Features = FE.reshape(-1, FE.shape[-1])
        Pred = RFc.predict(Features)
        Pred = np.reshape(Pred, xData.shape[:-1])
        Prediction.append(Pred)
        Time.Update((i+1)/len(XData))
    
    ImPred = np.array(Prediction)
    Time.Process(0)

    # Perform morphological operations
    for i, P in enumerate(ImPred):

        Pc = P.copy()

        # Remove small cement line regions
        Cl = P == 4
        Os = P == 2
        Threshold = 300
        Pad = np.pad(Cl+Os, ((1,1),(1,1)), mode='constant', constant_values=True)
        Regions = measure.label(Pad)
        Val, Co = np.unique(Regions, return_counts=True)
        Region = np.isin(Regions, Val[1:][Co[1:] > Threshold])
        Region = morphology.isotropic_erosion(Region, 1)
        Region = Region[1:-1,1:-1] * Cl
        P[(P == 4)*~Region] = 1
        P[(P == 4)*Region] = 4
        P[(P == 1)*~Region] = 1
        P[(P == 1)*Region] = 4

    # Compute metrics
    Prediction = ImPred.ravel()
    Precision = metrics.precision_score(Prediction, YData.ravel(), average=None)
    Recall = metrics.recall_score(Prediction, YData.ravel(), average=None)

    # Plot confusion matrix
    CMWeights = sWeights.reshape(-1, sWeights.shape[-1])
    CMWeights = np.max(CMWeights, axis=-1)
    
    CM = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize=None)
    Recall = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize='true')
    Precision = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize='pred')
    VSpace = 0.2/2
    Ticks = ['Background', 'Osteocyte\nLacuna', 'Haversian\nCanal', 'Cement\nLine']
    Figure, Axis = plt.subplots(1,1, figsize=(5.5,4.5), dpi=200)
    for Row in range(CM.shape[0]):
        for Column in range(CM.shape[1]):
            # Axis.text(x=Row, y=Column, position=(Row,Column), va='center', ha='center', s=CM[Row, Column])
            Axis.text(x=Row, y=Column, position=(Row,Column+VSpace), va='center', ha='center', s=round(Recall[Row, Column],2), color=(0,0,1))
            Axis.text(x=Row, y=Column, position=(Row,Column-VSpace), va='center', ha='center', s=round(Precision[Row, Column],2), color=(1,0,0))
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Ticks)),Ticks)
    Axis.set_yticks(np.arange(len(Ticks)),Ticks)
    Axis.set_ylim([-0.49,CM.shape[0]-0.5])
    Axis.set_xlim([-0.49,CM.shape[0]-0.5])
    Axis.set_xlabel('Recall',color=(0,0,1))
    Axis.set_ylabel('Precision',color=(1,0,0))
    Axis.xaxis.set_label_position('top')
    Axis.yaxis.set_label_position('right')
    plt.tight_layout()
    plt.savefig(str(ResultsDir / 'ConfusionMatrix.png'), dpi=200)
    plt.show()

    # Save predictions
    for i, P in enumerate(ImPred):

        S = P.shape
        Categories = utils.to_categorical(P)
        Seg = np.zeros((S[0],S[1],4))
        Seg[:,:,:-1] = Categories[:,:,2:] * 255
        BG = Categories[:,:,1].astype('bool')
        Seg[:,:,-1][~BG] = 200
        Seg[:,:,1][Seg[:,:,2] == 255] = 255

        DPI = 100
        IName = str(ResultsDir / 'Segmentation' / str('Seg_' + '%02d' % (i) + '.png'))
        Figure, Axis = plt.subplots(1,1, figsize=(S[0]/DPI, S[1]/DPI))
        Axis.imshow(ROIs[i])
        Axis.imshow(Seg.astype('uint8'), alpha=0.5)
        Axis.axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.savefig(IName, dpi=DPI)
        plt.close(Figure)

    # Visualize images that maximaze filter output
    for Layer in range(len(Outputs)):
        
        Dir = 'Layer_%02d' % Layer
        os.makedirs(ResultsDir / 'Unet' / Dir, exist_ok=True)

        NFilters = Outputs[Layer].shape[-1]
        Vz = Visualize(FeaturesExtractor, Layer=Layer)
        Vz.Show = False

        Images = []
        Time.Process(1,'Layer ' + str(Layer) + ' filters')
        for i in range(NFilters):
            Loss, Image = Vz.Filter(i)
            Images.append(Image)
            Time.Update((i+1) / NFilters)
        Time.Process(0)

        # Plot filters individually
        DPI = 100
        Time.Process(1,'Plot filters')
        for i, Image in enumerate(Images):
            S = Image.shape[:-1]
            FigName = ResultsDir / 'Unet' / Dir / ('Filter_%02d' % i)
            Figure, Axis = plt.subplots(1,1, figsize=(s/DPI for s in S),dpi=DPI)
            Axis.imshow(Image)
            Axis.axis('off')
            plt.subplots_adjust(0,0,1,1)
            plt.savefig(FigName,dpi=DPI)
            plt.close(Figure)
            Time.Update((i+1) / NFilters)
        Time.Process(0)

        # Plot filters on common figure
        C = 2
        R = 2
        DPI = 100
        FigName = ResultsDir / 'Unet' / Dir / Dir
        Figure, Axis = plt.subplots(R,C, figsize=(C*100/DPI,R*100/DPI))
        for c in range(C):
            for r in range(R):
                Axis[r,c].imshow(Images[r+c*R])
                Axis[r,c].axis('off')
        plt.subplots_adjust(0, 0, 1, 1, wspace=0.05, hspace=0.05)
        plt.savefig(FigName, dpi=DPI)
        plt.show(Figure)

    # Visualize example image
    Example = XData[0]
    FE = FeaturesExtraction(FeaturesExtractor, Example)
    Dir = ResultsDir / 'Unet' / 'Example'
    os.makedirs(Dir, exist_ok=True)
    DPI = 196
    Size = [S / DPI for S in FE.shape[:-1]]
    Shapes = [(1,3),(2,4),(2,4),(4,4),(4,4),(4,8),(4,8),
              (4,4),(4,4),(4,4),(2,4),(2,4),(2,4),(2,2)]
    i = 0
    for S in Shapes:
        Size = np.array(FE.shape[:-1]) * np.array(S) / DPI
        Figure, Axis = plt.subplots(S[0],S[1],figsize=[Size[1],Size[0]])
        if S[0] > 1:
            for j in range(S[0]):
                for k in range(S[1]):
                    Axis[j,k].imshow(FE[...,i], cmap='binary_r')
                    Axis[j,k].axis('off')
                    i += 1
        else:
            for k in range(S[1]):
                Axis[k].imshow(FE[...,i], cmap='binary_r')
                Axis[k].axis('off')
                i += 1
        plt.subplots_adjust(0,0,1,1, wspace=0.05, hspace=0.05)
        plt.savefig(str(Dir / ('Layer%03d'%i)), dpi=DPI)
        plt.close(Figure)

    # Show Unet and random forest predictions
    DPI = 196
    Figure, Axis = plt.subplots(1,1,figsize=(4.78,4.78),dpi=DPI)
    Axis.imshow(XData[0])
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(str(ResultsDir / 'Image'), dpi=DPI)
    plt.show(Figure)

    UPred = Unet_Probabilities(Unet, XData[0])
    Figure, Axis = plt.subplots(1,1,figsize=(4.78,4.78),dpi=DPI)
    Axis.imshow(UPred[:,:,1:])
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(str(ResultsDir / 'Unet_Prob'), dpi=DPI)
    plt.show(Figure)

    FE = FeaturesExtraction(FeaturesExtractor, XData[0])
    Features = FE.reshape(-1, FE.shape[-1])
    RFPred = RFc.predict_proba(Features)
    RFPred = np.reshape(RFPred, XData[0].shape[:2] +(4,))
    Figure, Axis = plt.subplots(1,1,figsize=(4.78,4.78),dpi=DPI)
    Axis.imshow(RFPred[:,:,1:])
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(str(ResultsDir / 'RFc_Prob'), dpi=DPI)
    plt.show(Figure)


    # Look at random forest features importances
    FI = RFc.feature_importances_
    Idx = np.argsort(FI)[::-1]
    CumSum = np.cumsum(FI[Idx])/sum(FI)

    Figure, Axis = plt.subplots(1,1)
    Axis.bar(np.arange(len(FI))+1, FI, width=0.8, color=(1,0,0))
    Axis.set_xlim([0,len(FI)+1])
    Axis.set_xlabel('Number (-)')
    Axis.set_ylabel('Feature Importance (-)')
    plt.savefig(str(ResultsDir / 'FeaturesImportance.png'), dpi=200)
    plt.show(Figure)

    Figure, Axis = plt.subplots(1,1)
    Axis2 = Axis.twinx()
    Axis.plot(CumSum, color=(1,0,0))
    Axis2.plot(FI[Idx]/max(FI), color=(0,0,1))
    Axis.set_ylabel('Cumulative Sum (-)', color=(1,0,0))
    Axis2.set_ylabel('Relative Importance (-)', color=(0,0,1))
    plt.savefig(str(ResultsDir / 'FeaturesImportance2.png'), dpi=200)
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
