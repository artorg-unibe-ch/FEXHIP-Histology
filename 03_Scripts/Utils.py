"""
This script provides utility functions

Version: 01

Author: Mathieu Simon
        ARTORG Center for Biomedical Engineering Research
        SITEM Insel
        University of Bern

Date: October 2023
"""

import os
import time
import numpy as np
from pathlib import Path
from patchify import patchify
from skimage import color, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import Model, utils

# Main functions
def StainNA(Image:np.array, Mean:np.array, Std:np.array) -> np.array:

    """
    Perform stain normalization according to the method described here:
        Shen et al. 2022
        RandStainNA: Learning Stain-Agnostic Features from Histology
        Slides by Bridging Stain Augmentation and Normalization

    :param Image: 3d (2d + RGB) numpy array
    :param  Mean: (3,) shaped numpy array containing mean of reference
                  image in LAB space
    :param   Std: (3,) shaped numpy array containing standard deviation
                  of reference image in LAB space
    :return Norm: 3d (2d + RGB) numpy array of the stain normalized image
    """
    
    LAB = color.rgb2lab(Image)
    LAB_Mean = np.mean(LAB, axis=(0,1))
    LAB_Std = np.std(LAB, axis=(0,1), ddof=1)

    Norm = (LAB - LAB_Mean) / LAB_Std * Std + Mean

    RGB = color.lab2rgb(Norm)

    return RGB

def Unet_Probabilities(Unet:Model, Image:np.array, Margin=10) -> np.array:

    """
    Compute probabilities of each pixel class using keras U-net model
    Pad the original image to fit a multiple of U-net input shape
    Patchify padded image with patches fitting the U-net input shape
    Perform U-net prediction for each patch
    Rescale prediction to the 0-1 range

    :param    Unet: Keras U-net model
    :param   Image: 3d (2d + RGB) numpy array with each channel in 0-1
                    range
    :param  Margin: Integer giving the minimum margin to remove from
                    U-Net prediction as it gets less accurate at the
                    borders
    :return Scaled: 3d (2d + classes) numpy array of probabilities for
                    each pixel class in 0-1 range
    """
            
    Size = np.array(Unet.input_shape[1:-1])
    Step = np.concatenate([Size - 2*Margin, [Unet.input_shape[-1]]])
    
    # Compute necessary number of patches
    NPatches = np.ceil(np.array(Image.shape)[:-1] / Step[:-1]).astype(int)

    # Pad image to fit patch size, step size and respect margin
    ISize = NPatches * Step[:-1] + 2*Margin
    Pad = (ISize - np.array(Image.shape[:-1])) // 2
    Padded = np.pad(Image, [[Pad[0],Pad[0]],[Pad[1],Pad[1]],[0,0]], mode='reflect')

    # Separate image into patches to fit U-net
    Size = np.concatenate([Size, [3]])
    Patches = patchify(Padded, Size, step=Step)

    # Get U-net probabilities
    Prob = np.zeros(np.concatenate([Padded.shape[:-1], [4]]), float)
    for Xi, Px in enumerate(Patches):
        for Yi, Py in enumerate(Px):
            Pred = Unet.predict(Py, verbose=0)[0]
            X1 = Xi*Step[0] + Margin
            X2 = Step[0] + Xi*Step[0] + Margin
            Y1 = Yi*Step[1] + Margin
            Y2 = Step[1] + Yi*Step[1] + Margin
            Prob[X1:X2, Y1:Y2] += Pred[Margin:-Margin,Margin:-Margin]

    return Prob[Pad[0]:-Pad[0],Pad[1]:-Pad[1]]

def FeaturesExtraction(FE:Model, Image:np.array, Margin=10) -> np.array:

    """
    Extract features using unet outputs
    Rescale prediction to the 0-1 range

    :param    Unet: Keras U-net model
    :param   Image: 3d (2d + RGB) numpy array with each channel in 0-1
                    range
    :param  Margin: Integer giving the minimum margin to remove from
                    U-Net prediction as it gets less accurate at the
                    borders
    :return Scaled: 3d (2d + classes) numpy array of probabilities for
                    each pixel class in 0-1 range
    """
            
    Size = np.array(FE.input_shape[1:-1])
    Step = np.concatenate([Size - 2*Margin, [FE.input_shape[-1]]])
    
    # Compute necessary number of patches
    NPatches = np.ceil(np.array(Image.shape)[:-1] / Step[:-1]).astype(int)

    # Pad image to fit patch size, step size and respect margin
    ISize = NPatches * Step[:-1] + 2*Margin
    Pad = (ISize - np.array(Image.shape[:-1])) // 2
    Padded = np.pad(Image, [[Pad[0],Pad[0]],[Pad[1],Pad[1]],[0,0]], mode='reflect')

    # Separate image into patches to fit U-net
    Size = np.concatenate([Size, [3]])
    Patches = patchify(Padded, Size, step=Step)

    # Get U-net probabilities
    Repeat = [2, 2, 4, 4, 2, 2, 2]
    NFeatures = 3 + sum([F[-1] for F in FE.output_shape])
    Features = np.zeros(Padded.shape[:-1] + (NFeatures,), float)
    for Xi, Px in enumerate(Patches):
        for Yi, Py in enumerate(Px):
            FeaturesList = FE.predict(Py, verbose=0)
            X1 = Xi*Step[0] + Margin
            X2 = Step[0] + Xi*Step[0] + Margin
            Y1 = Yi*Step[1] + Margin
            Y2 = Step[1] + Yi*Step[1] + Margin
            Features[X1:X2, Y1:Y2, :3] += Py[0][Margin:-Margin,Margin:-Margin]
            F1, F2 = 3, 3
            for Fi, F in enumerate(FeaturesList):
                F2 += F.shape[-1]
                if 9 > Fi > 1:
                    F = np.repeat(F, Repeat[Fi-2], axis=1)
                    F = np.repeat(F, Repeat[Fi-2], axis=2)
                Features[X1:X2, Y1:Y2, F1:F2] += F[0][Margin:-Margin,Margin:-Margin]
                F1 = F2

    return Features[Pad[0]:-Pad[0],Pad[1]:-Pad[1]]


#%% Time
# Time functions
class Time():

    """
    Class for time measuring and printing functions
    """

    def __init__(self, Width=15, Length=21, Text='Process') -> None:

        """
        Initialize Time class
        :param  Width: Width of the 0-1 progress printing range
        :param Length: Maximum text lenght
        :param   Text: Text to print, process name
        :return  None
        """

        self.Width = Width
        self.Length = Length
        self.Text = Text
        self.Tic = time.time()
        
        return
    
    def Set(self, Tic=None) -> None:

        """
        Set a starting time point to measure time
        :param Tic: Specific starting time point
        """
        
        if Tic == None:
            self.Tic = time.time()
        else:
            self.Tic = Tic

        return

    def Print(self, Tic=None,  Toc=None) -> None:

        """
        Print elapsed time in seconds to time in HH:MM:SS format
        :param   Tic: Actual time at the beginning of the process
        :param   Toc: Actual time at the end of the process
        :return None
        """

        if Tic == None:
            Tic = self.Tic
            
        if Toc == None:
            Toc = time.time()


        Delta = Toc - Tic

        Hours = np.floor(Delta / 60 / 60)
        Minutes = np.floor(Delta / 60) - 60 * Hours
        Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

        print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

        return

    def Update(self, Progress, Text='') -> None:

        """
        Function used to print the progress of the process
        :param Progress: Float in 0-1 range representing
                         the progress of the process
        :param     Text: String to print for the process state
        :return    None
        """

        Percent = int(round(Progress * 100))
        Np = self.Width * Percent // 100
        Nb = self.Width - Np

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        Ns = self.Length - len(Text)
        if Ns >= 0:
            Text += Ns*' '
        else:
            Text = Text[:self.Length]
        
        Line = '\r' + Text + ' [' + Np*'=' + Nb*' ' + ']' + f' {Percent:.0f}%'
        print(Line, sep='', end='', flush=True)

        return

    def Process(self, StartStop:bool, Text='') -> None:

        """
        Function used to measure and print time elapsed for
        a specific process
        :param StartStop: Boolean value, 1 to start; 0 to stop
        :param      Text: Text to print
        :return     None
        """

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        if StartStop*1 == 1:
            print('')
            self.Set()
            self.Update(0, Text)

        elif StartStop*1 == 0:
            self.Update(1, Text)
            self.Print()
        
        return
Time = Time()

#%% CVAT
# CVAT functions
class CVAT():

    """
    List of functions used to analyse, manipulate data
    generated using CVAT
    """

    def __init__(self:None) -> None:
        pass

    def GetData(self, Path:Path):

        """
        Collect data generated manually using CVAT
        :param Path: Path to main directory of CVAT data
        :return   Name: ROI names
                Images: Original ROI
                Labels: Manual segmentation performed on CVAT
        """

        Time.Process(1,'Collect CVAT data')

        Operators = sorted([D for D in Path.iterdir() if D.is_dir()])

        Images = []
        Names = []
        Labels = []
        for i, Operator in enumerate(Operators):

            # List images name
            FPath = Operator / 'data'
            Files = [F for F in FPath.iterdir() if F.name.endswith('.png')]
            Names.append([F.name[:-4] for F in Files])

            # Collect images
            Images.append([io.imread(F)[:,:,:-1] for F in Files])

            # Collect masks
            FPath = Operator / 'segmentation' / 'SegmentationClass'
            Labels.append([io.imread(FPath / F.name) for F in Files])

            Time.Update((i+1)/len(Operators))
        Time.Process(0)

        self.Names = np.array(Names)
        self.Images = np.array(Images)
        self.Labels = np.array(Labels)

        return np.array(Names), np.array(Images), np.array(Labels)
        
    def OneHot(self, Labels:np.array([])) -> np.array:

        """
        Perform one-hot encoding of masks generated with CVAT
        :param Labels: List of segmentation masks generated
                    with CVAT
        :return OneHot: One-hot encoding of the labels
        """

        Time.Process(1,'Perform encoding')

        if Labels.size == 0:
            Labels = self.Labels

        L = Labels[0,0]
        Colors = np.unique(L.reshape(-1, L.shape[2]), axis=0)
        
        OneHot = np.zeros((Labels.shape[:-1] + (len(Colors),)))
        for i, C in enumerate(Colors):
            OneHot[...,i] = np.all(Labels == C, axis=-1)
            Time.Update((i+1)/len(Colors))
        Time.Process(0)

        return np.array(OneHot * 1, int)

    def CommonROI(self):

        """
        Determine common ROI, name, and label
        Compute it's segmentation ratio as one-hot encoding
        """

        Uniques, Counts = np.unique(self.Names, return_counts=True)
        CommonName = Uniques[np.argmax(Counts)]
        Indices = self.Names == CommonName
        CommonROI = self.Images[0][Indices[0]][0]

        Expanded = np.expand_dims(self.Labels[Indices], 1)
        OneHot = self.OneHot(Expanded)
        Sum = np.sum(OneHot, axis=(0,1))
        SegRatio = Sum / Sum.max()

        return CommonName, CommonROI, SegRatio, Indices

    def SampleWeights(self, OneHot:np.array, Indices:list, SegRatio:np.array) -> np.array:

        """
        Compute sample weights according to segmentation ratio
        of common ROI
        """
        Time.Process(1, 'Compute sample weights')
        sWeights = np.zeros(OneHot.shape)
        for i, OH in enumerate(OneHot):
            Index = np.argmax(Indices[i])
            Ratio = OH[Index] * SegRatio
            Ratio[Ratio == 0.0] = np.nan
            Values = np.nanmean(Ratio, axis=(0,1))

            for Idx in range(len(Indices[i])):
                sWeights[i,Idx] = OneHot[i,Idx] * Values
            Time.Update((i+1)/len(OneHot))

        # Set background sample weights as mean operator score
        F = OneHot[...,0] == 1
        sWeights = sWeights[...,1:].sum(axis=-1)
        mWeights = np.ma.masked_array(sWeights, F)
        Means = np.ma.mean(mWeights,axis=(-1,-2)).data
        Means = np.expand_dims(Means, (-1,-2))
        Means = F * Means
        sWeights[F] = Means[F]

        Time.Process(0)

        return sWeights
CVAT = CVAT()

#%% Training
# Model training functions
class Training():

    def __init__(self:None) -> None:
        pass

    def GetData(self, Path:Path):

        """
        To write
        """

        Time.Process(1,'Get training data')
        Data = sorted([F for F in Path.iterdir()])

        Images = []
        Labels = []
        for i, File in enumerate(Data):

            if File.name.startswith('ROI'):
                Images.append(io.imread(File))
            else:
                Labels.append(io.imread(File))

            Time.Update((i+1)/len(Data))
        Time.Process(0)

        return np.array(Images), np.array(Labels)
    
    def SetWeights(self, Labels:np.array):

        """
        To write
        """

        sWeights = Labels / 255
        YData = np.zeros((sWeights.shape[:-1] + (4,)), int)
        F1 = sWeights[:,:,:,2] > 0
        YData[F1] = [0, 0, 0, 1]
        F2 = sWeights[:,:,:,1] > 0
        YData[F2] = [0, 0, 1, 0]
        F3 = sWeights[:,:,:,0] > 0
        YData[F3] = [0, 1, 0, 0]
        YData[~(F1+F2+F3)] = [1, 0, 0, 0]

        # Set background sample weights as mean operator score
        sWeights = sWeights.sum(axis=-1)
        mWeights = np.ma.masked_array(sWeights, ~(F1+F2+F3))
        Means = np.ma.mean(mWeights,axis=(-1,-2)).data
        Means = np.expand_dims(Means, (-1,-2))
        Means = ~(F1+F2+F3) * Means
        sWeights[~(F1+F2+F3)] = Means[~(F1+F2+F3)]

        return sWeights, YData

Training = Training()

#%% Reference
# Reference image stats
class Reference():

    """
    Class for references values in the LAB space
    """

    def __init__(self:None,
                 Mean=np.array([54.26,  11.97, -31.79]),
                 Std=np.array([11.07,  5.32, 11.80])
                 ) -> None:
        
        """
        Initialize Reference class with LAB channels mean and
        standard deviation values
        :param  Mean: Mean of each of the LAB channel
        :param   Std: Standard deviation of each of the LAB channel
        :return None
        """

        self.Mean = Mean
        self.Std = Std

        return

    def SetNew(self:None, Image:np.array) -> None:

        """
        Compute stats of new reference image
        :param Image: 3d (2d + RGB) numpy array
        """

        LAB = color.rgb2lab(Image)
        self.Mean = np.mean(LAB, axis=(0,1))
        self.Std = np.std(LAB, axis=(0,1), ddof=1)

        return
Reference = Reference()