#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script aims to preform preprocessing on manually segmented
    images which will serve as training data for the automatic 
    segmentation model

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: February 2024
    """

#%% Imports
# Modules import

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from matplotlib.colors import ListedColormap

from Utils import Time, Reference, StainNA, CVAT

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
def OperatorVar(Var:pd.DataFrame) -> None:

    """
    Compute and plot inter-operator variability
    """

    Cols = Var.columns
    Colors = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]

    # Plot 
    Figure, Axis = plt.subplots(2,2, sharex=True, sharey=True, dpi=200)
    for i in range(2):
        for ii in range(2):
            j = 2*i+ii
            for Array in Var[Cols[j]]:
                Values, Count = np.unique(Array, return_counts=True)
                Axis[i,ii].bar(Values, Count / sum(Count),
                               width=(Values[1]-Values[0])*0.8,
                               color=Colors[j] + (1/len(Var),))
                Axis[i,ii].set_title(Cols[j])
                Axis[i,ii].set_ylim([0,1.05])
    for i in range(2):
        Axis[1,i].set_xlabel('Agreement Ratio (-)')
        Axis[i,0].set_ylabel('Relative pixels number (-)')
    plt.show(Figure)


    # Plot 
    Figure, Axis = plt.subplots(2,2, sharex=True, sharey=True, dpi=200)
    for i in range(2):
        for ii in range(2):
            j = 2*i+ii
            for k, Array in enumerate(Var[Cols[j]]):
                k += 1
                Std = np.std(Array, ddof=1)
                Mean = np.mean(Array)
                Axis[i,ii].plot(k,Mean,color=Colors[j],marker='o')
                Axis[i,ii].plot([k,k],[Mean-Std, Mean+Std],
                                color=Colors[j], marker='_', mew=1.5)
                Axis[i,ii].set_title(Cols[j])
                Axis[i,ii].set_ylim([0,1.05])
    for i in range(2):
        Axis[1,i].set_xlabel('Operator Number (-)')
        Axis[1,i].set_xticks(np.arange(k+2)[2::2])
        Axis[i,0].set_ylabel('Agreement Ratio (-)')
    plt.show(Figure)

    return
def PlotDistribution(Data:list, Variable='') -> None:

    """
    Plot distribution (histogram) of list of data
    """
        
    # Get data variables
    SortedValues = np.sort(Data)
    N = len(Data)
    X_Bar = np.mean(Data)
    S_X = np.std(Data, ddof=1)

    KernelEstimator = np.zeros(N)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25,0.75]), 0, 1)))
    DataIQR = abs(np.abs(np.quantile(Data,0.75)) - np.abs(np.quantile(Data,0.25)))
    KernelHalfWidth = 0.9*N**(-1/5) * min(S_X,DataIQR/NormalIQR)
    for Value in SortedValues:
        KernelEstimator += norm.pdf(SortedValues-Value,0,KernelHalfWidth*2)
    KernelEstimator = KernelEstimator/N

    # Histogram and density distribution
    TheoreticalDistribution = norm.pdf(SortedValues,X_Bar,S_X)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.hist(Data,density=True,bins=20,edgecolor=(0,0,1),color=(1,1,1),label='Histogram')
    Axes.plot(SortedValues,KernelEstimator,color=(1,0,0),label='Kernel Density')
    Axes.plot(SortedValues,TheoreticalDistribution,linestyle='--',color=(0,0,0),label='Normal Distribution')
    plt.xlabel(Variable)
    plt.ylabel('Density (-)')
    plt.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.5,1.15), prop={'size':10})
    plt.show()

    return
def DataStats(Images:list, Plots=False) -> tuple:

    """
    Compute average and standard deviation of each canal
    for a list of images in the LAB space. Then, fit a
    normal distribution to these values and return the
    resulting parameters into two arrays
    """

    # Unfold Images list
    Images = [ii for i in Images for ii in i]
    Averages = np.zeros((len(Images),3))
    Stds = np.zeros((len(Images),3))

    # Store average and standard deviation in lab space
    for i, Image in enumerate(Images):
        LAB = color.rgb2lab(Image)
        Averages[i] = np.mean(LAB, axis=(0,1))
        Stds[i] = np.std(LAB, axis=(0,1), ddof=1)

    # Plot distribution
    if Plots:
        PlotDistribution(Averages[:,0],'L')
        PlotDistribution(Averages[:,1],'A')
        PlotDistribution(Averages[:,2],'B')

        PlotDistribution(Stds[:,0],'L')
        PlotDistribution(Stds[:,1],'A')
        PlotDistribution(Stds[:,2],'B')

    # Fit normal distribution
    aL = norm.fit(Averages[:,0])
    aA = norm.fit(Averages[:,1])
    aB = norm.fit(Averages[:,2])

    sL = norm.fit(Stds[:,0])
    sA = norm.fit(Stds[:,1])
    sB = norm.fit(Stds[:,2])
            
    return (np.array([aL, aA, aB]), np.array([sL, sA, sB]))
def StainAugmentation(Images:list, aLAB:np.array, sLAB:np.array, N:int) -> np.array:

    """
    Perform stain augmentation according to
        Shen et al. 2022
        RandStainNA: Learning Stain-Agnostic Features from Histology
        Slides by Bridging Stain Augmentation and Normalization
    """

    # Convert images into LAB space
    LAB = []
    S = Images[0].shape
    Time.Process(1,'Convert to LAB')
    for I in Images:
        lab = color.rgb2lab(I / 255)
        Nlab = []
        for _ in range(N):
            Nlab.append(lab)
        LAB.append(Nlab)
        Time.Update(len(LAB)/3/len(Images))
    LAB = np.array(LAB)

    # Define random transform parameters (reshape first to match data shape)
    Time.Update(1/3, 'Tranform Images')
    aLAB = np.tile(aLAB.T, N).reshape((2, N, 3), order='A')
    aLAB = np.tile(aLAB, len(Images)).reshape((2, len(Images), N, 3))
    sLAB = np.tile(sLAB.T, N).reshape((2, N, 3), order='A')
    sLAB = np.tile(sLAB, len(Images)).reshape((2, len(Images), N, 3))
    Mean = np.random.normal(aLAB[0], aLAB[1])
    Std = np.random.normal(sLAB[0], sLAB[1])

    # Normalize images according to random templates
    S = LAB.shape
    Norm = np.zeros(S)
    X_Bar = np.mean(LAB, axis=(2,3))
    S_X = np.std(LAB, axis=(2,3), ddof=1)

    X_Bar = np.tile(X_Bar, S[2]*S[3]).reshape(S)
    S_X = np.tile(S_X, S[2]*S[3]).reshape(S)
    Mean = np.tile(Mean, S[2]*S[3]).reshape(S)
    Std = np.tile(Std, S[2]*S[3]).reshape(S)

    Norm = (LAB - X_Bar) / S_X * Std + Mean

    # Convert back to RGB space
    Results = []
    Time.Update(2/3, 'Convert to RGB')
    for i in range(len(Images)):
        results = []
        for j in range(N):
            RGB = color.lab2rgb(Norm[i,j])
            results.append(RGB)
        Results.append(results)
        Time.Update(2/3 + i/3/len(Images))
    Results = np.array(Results) * 255

    Time.Process(0)

    return np.round(Results).astype('uint8')
def DataAugmentation(StainAug:np.array, Masks:np.array, ResultsPath:Path, N=8, ROISize=256) -> None:

    """
    Perform data augmentation and save results to directory
    :param    StainAug: Stain augmented data
    :param       Masks: Labels
    :param ResultsPath: Directory to save results
    :param           N: Augmentation factor
    :param     ROISize: Size of final ROI
    """


    Time.Process(1,'Augment Data')
    Path.mkdir(ResultsPath, exist_ok=True)
    Size = StainAug.shape[0] * StainAug.shape[1]
    for s, Stains in enumerate(StainAug):
        for i, Image in enumerate(Stains):
            for n in range(N):

                # Flip image
                if np.mod(n,2)-1:
                    fImage = Image
                    fLabel = Masks[s]
                elif np.mod(n,4)-1:
                    fImage = Image[::-1, :, :]
                    fLabel = Masks[s][::-1, :]
                else:
                    fImage = Image[:, ::-1, :]
                    fLabel = Masks[s][:, ::-1]

                # Rotate image of 90 degrees
                if n < 2:
                    rImage = fImage
                    rLabel = fLabel
                elif n < 4:
                    rImage = np.rot90(fImage,1)
                    rLabel = np.rot90(fLabel,1)
                elif n < 6:
                    rImage = np.rot90(fImage,2)
                    rLabel = np.rot90(fLabel,2)
                else:
                    rImage = np.rot90(fImage,3)
                    rLabel = np.rot90(fLabel,3)

                # Select random location
                Min = 0
                Max = rImage.shape[0] - ROISize
                X0, Y0 = np.random.randint(Min, Max, 2)
                ROI = rImage[X0:X0+ROISize, Y0:Y0+ROISize]
                Lab = rLabel[X0:X0+ROISize, Y0:Y0+ROISize]

                # Convert to uint8
                Lab = np.round(Lab*255).astype('uint8')

                # Save augmented image and label
                Name = 'ROI_%02d_%1d_%1d.png' % (s, i, n)
                io.imsave(ResultsPath / Name, ROI)
                Name = 'Lab_%02d_%1d_%1d.png' % (s, i, n)
                io.imsave(ResultsPath / Name, Lab, check_contrast=False)

        Time.Update((s+1) * (i+1) / Size)
    Time.Process(0)

    return


#%% Main
# Main part

def Main():

    # List training data
    Dirs = SetDirectories('FEXHIP-Histology')
    TrainingPath = Dirs['Data'] / 'Training'
    Names, Images, Labels = CVAT.GetData(TrainingPath)

    # Collect individual segments as one-hot encoding
    OneHot = CVAT.OneHot(Labels)

    # Get common ROI
    CommonName, CommonROI, SegRatio, Indices = CVAT.CommonROI()

    # Get interoperator variability
    Cols = ['Interstitial Tissue', 'Osteocytes', 'Haversian Canals', 'Cement Lines']
    Var = pd.DataFrame(index=np.arange(len(OneHot)), columns=Cols)
    for i, OH in enumerate(OneHot):
            Index = np.argmax(Indices[i])
            Ratio = OH[Index] * SegRatio
            for j in range(4):
                Val = Ratio[:,:,j]
                F = OH[Index][:,:,j]
                Var.loc[i, Cols[j]] = Val[F > 0]

    OperatorVar(Var)

    # Plot common ROI
    Figure, Axis = plt.subplots(3,4)
    for i in range(3):
        for j in range(4):
            Index = np.argmax(Indices[j+4*i])
            Axis[i,j].set_title('Operator ' + str(j+4*i+1))
            Axis[i,j].imshow(OneHot[j+4*i,Index,:,:,1:]*255)
            Axis[i,j].axis('off')
    plt.show(Figure)

    # Compute sample weights
    sWeights = CVAT.SampleWeights(OneHot, Indices, SegRatio)

    # Keep only 1 occurence of common ROI
    # Use agreement ratio as sample weight
    ROIs = Images[~Indices]
    Masks = np.expand_dims(sWeights[~Indices],-1) * OneHot[~Indices]
    ROIs = np.concatenate([np.expand_dims(CommonROI, 0), ROIs])
    Masks = np.concatenate([np.expand_dims(SegRatio, 0), Masks])

    # Custom color map
    N = 256
    CValues = np.zeros((N, 4))
    CValues[:, 0] = np.linspace(0, 1, N)
    CValues[:, 1] = np.linspace(1, 0, N)
    CValues[:, 2] = np.linspace(1, 0, N)
    CValues[:, -1] = np.linspace(1.0, 1.0, N)
    CMP = ListedColormap(CValues)

    # Plot agreement ratio
    SegRatio[SegRatio == 0.0] = np.nan
    Figure, Axis = plt.subplots(1,3, dpi=500)
    for i in range(3):
        Axis[i].imshow(CommonROI)
        Plot = Axis[i].imshow(SegRatio[:,:,i+1], cmap=CMP)
        Axis[i].axis('off')
    CBarAxis = Figure.add_axes([0.2, 0.25, 0.6, 0.025])
    plt.colorbar(Plot, cax=CBarAxis, orientation='horizontal', label='Segmentation ratio (-)')
    plt.show(Figure)

    # Perform stain normalization
    Norms = []
    for ROI in ROIs:
        Norm = StainNA(ROI, Reference.Mean, Reference.Std)
        Norms.append(Norm)

    # Stain augmentation
    N = 2
    AverageLAB, StdLAB = DataStats(Images)
    AverageLAB[:,0] = Reference.Mean
    StdLAB[:,0] = Reference.Std
    StainAug = StainAugmentation(Norms, AverageLAB, StdLAB, N)
    
    # Perform data augmentation
    ResultsPath =  Dirs['Results'] / '01_Training' / 'Preprocessing'
    os.makedirs(ResultsPath, exist_ok=True)
    DataAugmentation(StainAug, Masks[:,:,:,1:], ResultsPath)

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
