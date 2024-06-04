#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script performs analysis of segmentation results

    Detailed description:
        Simon et al. (2024)
        Automatic Segmentation of Cortical Bone Structure
        SomeJournal, x(x), xxx-xxx
        https://doi.org/

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: October 2023
    """

#%% Imports
# Modules import

import argparse
import numpy as np
import pandas as pd
from dom import DOM
from Utils import Time
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage import io, measure, morphology
from scipy.ndimage import distance_transform_edt as Euclidian

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

def PlotExample(Mask:np.array, Dist:np.array, Lines:np.array, FName:str, Phantom=np.array([])) -> None:

    """
    Plot an example of a segmentation (figure for article)
    """

    # Diaphysis_Left_Zoom_06.png

    Size = Mask.shape
    FigureMask = np.zeros((Size + (4,)),'uint8')
    FigureMask[Mask] = [255, 255, 255, 255]
    FigureMask[Lines == 0] = [255, 0, 0, 255]
    FigureMask[Phantom] = [0, 0, 0, 255]


    DPI = 192
    Figure, Axis = plt.subplots(1,1, figsize=np.array(Size)/DPI)
    Axis.imshow(Dist)
    Axis.imshow(FigureMask)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(FName, dpi=DPI)
    plt.show(Figure)

    return

#%% Main
# Main part

def Main():

    # Record elapsed time
    Time.Process(1, 'Collect Seg Data')

    # Set paths
    Dirs = SetDirectories('FEXHIP-Histology')
    DataDir = Dirs['Results'] / '03_Segmentation'
    ResDir = Dirs['Results'] / '04_Statistics'

    # List folders and create data frame
    Folders = sorted([F for F in DataDir.iterdir() if Path.is_dir(F)])
    Folders = Folders[:-1]
    Products = [np.arange(1,len(Folders)+1),
                ['Right','Left'],
                ['Diaphysis','Neck Inferior','Neck Superior'],
                np.arange(1,41)]
    Indices = pd.MultiIndex.from_product(Products, names=['Donor','Side','Site','ROI'])
    Tissues = ['Haversian canals', 'Osteocytes', 'Cement lines']
    Variables = ['Density (%)', 'Area (um2)', 'Number (-)', 'L1 (um)', 'L2 (um)',
                 'Aspect Ratio (-)', 'Characteristic Length (um)', 'Thickness (um)']
    Columns = pd.MultiIndex.from_product([Tissues, Variables])
    Data = pd.DataFrame(index=Indices, columns=Columns)

    # Pixel spacing
    PS = 1.046

    # Store distances distribution
    NMax = 500
    HCHC_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    HCOS_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    HCCL_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    OSOS_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    HC_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    OS_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)
    CL_Distribution = pd.DataFrame(index=Indices, columns=np.arange(NMax)+1)

    # Store osteocytes color distribution
    RGBColumns = pd.MultiIndex.from_product([['R','G','B'], np.arange(255)+1])
    OS_Colors = pd.DataFrame(index=Indices, columns=RGBColumns)

    # Store image sharpness and image channel levels
    SharpMeasure = DOM()
    ImageQuality = pd.DataFrame(index=Indices, columns=RGBColumns)
    ImageQuality['Sharpness'] = np.nan
    ImageQuality = pd.read_csv(ResDir / 'ImageQuality.csv', index_col=[0,1,2,3], header=[0,1])


    # Loop over every donor, every sample
    for iF, Folder in enumerate(Folders):

        # List and loop over ROI images
        Files = sorted([F for F in Folder.iterdir()])
        for File in Files:

            # Set index
            Name = File.name.split('_')
            if len(Name) == 5:
                Site = Name[0] + ' ' + Name[2] + 'erior'
            else:
                Site = Name[0]
            Index = (int(Folder.name[-2:]),
                     Name[1],
                     Site,
                     int(Name[-1][:-4]) + 1)

            # Read segmented image and original ROI
            Mask = io.imread(File)
            Image = io.imread(str(File).replace('03_Segmentation','02_ROIs'))

            # Store image rgb histograms
            ImageQuality.loc[Index,'R'] = np.histogram(Image[...,0], bins=255, range=[0,255])[0]
            ImageQuality.loc[Index,'G'] = np.histogram(Image[...,1], bins=255, range=[0,255])[0]
            ImageQuality.loc[Index,'B'] = np.histogram(Image[...,2], bins=255, range=[0,255])[0]

            # Compute image sharpness
            Sharpness = SharpMeasure.get_sharpness(str(File).replace('03_Segmentation','02_ROIs'))
            ImageQuality.loc[Index,'Sharpness'] = Sharpness

            # Get individual masks
            OS = Mask[:,:,0] == 255
            HC = Mask[:,:,1] == 255
            CL = Mask[:,:,2] == 255

            # Compute distances
            HC_Dist = Euclidian(1-HC) * PS
            OS_Dist = Euclidian(1-OS) * PS
            CL_Dist = Euclidian(1-CL) * PS

            # Compute amount of bone of a given distance of a structure
            HC_Hist = np.histogram(HC_Dist, bins=NMax, range=[0,NMax])[0]
            HC_Distribution.loc[Index] = np.cumsum(HC_Hist)/np.sum(HC_Hist)

            OS_Hist = np.histogram(OS_Dist[1-HC], bins=NMax, range=[0,NMax])[0]
            OS_Distribution.loc[Index] = np.cumsum(OS_Hist)/np.sum(OS_Hist)
            
            CL_Hist = np.histogram(CL_Dist[1-HC], bins=NMax, range=[0,NMax])[0]
            CL_Distribution.loc[Index] = np.cumsum(CL_Hist)/np.sum(CL_Hist)

            # Get max distances
            HC_Markers = measure.label(HC)
            OS_Markers = measure.label(OS)
            HC_WS = watershed(HC_Dist, HC_Markers, watershed_line=True)
            OS_WS = watershed(OS_Dist, OS_Markers, watershed_line=True)

            # Plot example
            PlotExample(OS, HC_Dist, HC_WS, 'Test2', HC)

            # Remove Haversian canals from max distances
            OS_WS[HC] += 1

            FName = Dirs['CWD'] / '05_Presentations' / 'Analysis' / 'HCDistances'
            PlotExample(HC, HC_Dist, HC_WS, FName)
            FName = Dirs['CWD'] / '05_Presentations' / 'Analysis' / 'OSDistances'
            PlotExample(OS, OS_Dist, OS_WS, FName, HC)
            FName = Dirs['CWD'] / '05_Presentations' / 'Analysis' / 'CLDistances'
            PlotExample(CL, CL_Dist, OS_WS+1, FName, HC)

            # Inter-Haversian canals and inter-osteocytes distances
            HCHC_Distances = 2 * HC_Dist[HC_WS == 0]
            OSOS_Distances = 2 * OS_Dist[OS_WS == 0]

            # If more than 1 Haversian canal, store inter HC distances
            if len(HCHC_Distances) > 0:
                HC_Hist = np.histogram(HCHC_Distances, bins=NMax, range=[0,NMax])[0]
                HCHC_Distribution.loc[Index] = np.cumsum(HC_Hist)/np.sum(HC_Hist)
            
            # Store inter-osteocytes distances
            OS_Hist = np.histogram(OSOS_Distances, bins=NMax, range=[0,NMax])[0]
            OSOS_Distribution.loc[Index] = np.cumsum(OS_Hist)/np.sum(OS_Hist)

            # Osteocytes to Haversian canals distances
            HCOS_Distances = HC_Dist[OS]
            OS_Hist = np.histogram(HCOS_Distances, bins=NMax, range=[0,NMax])[0]
            HCOS_Distribution.loc[Index] = np.cumsum(OS_Hist)/np.sum(OS_Hist)

            # Cement lines to Haversian canals distances
            HCCL_Distances = HC_Dist[CL]
            CL_Hist = np.histogram(HCCL_Distances, bins=NMax, range=[0,NMax])[0]
            HCCL_Distribution.loc[Index] = np.cumsum(CL_Hist)/np.sum(CL_Hist)

            # Measure osteocytes area
            Labels = measure.label(OS)
            Props = ['area','axis_major_length','axis_minor_length']
            RP = measure.regionprops_table(Labels, properties=Props)
            RP = pd.DataFrame(RP)
            RP['Aspect Ratio (-)'] = RP[Props[1]] / RP[Props[2]]

            # Store into data frame
            Data.loc[Index]['Osteocytes','Area (um2)'] = RP['area'].mean()*PS**2
            Data.loc[Index]['Osteocytes','Number (-)'] = len(RP)
            Data.loc[Index]['Osteocytes','L1 (um)'] = RP[Props[1]].mean()*PS
            Data.loc[Index]['Osteocytes','L2 (um)'] = RP[Props[2]].mean()*PS
            Data.loc[Index]['Osteocytes','Aspect Ratio (-)'] = RP['Aspect Ratio (-)'].mean()
            Data.loc[Index]['Osteocytes','Characteristic Length (um)'] = np.sqrt(RP['area'].mean()*PS**2)
            
            # Get individual osteocytes colors distribution
            RGB = np.zeros((3,255))
            for L in np.unique(Labels)[1:]:
                for i in range(3):
                    RGB[i] += np.histogram(Image[Labels == L][:,i], bins=255, range=[0,255])[0]
            RGB = RGB / np.sum(RGB, axis=1)[0]

            OS_Colors.loc[Index, 'R'] = RGB[0]
            OS_Colors.loc[Index, 'G'] = RGB[1]
            OS_Colors.loc[Index, 'B'] = RGB[2]

            # Measure Haversian canals not connected to border
            Props = ['area', 'axis_major_length', 'axis_minor_length', 'label']
            Labels = measure.label(HC)
            RP1 = measure.regionprops_table(Labels, properties=Props)
            RP1 = pd.DataFrame(RP1)
            RP2 = measure.regionprops_table(Labels[1:-1,1:-1], properties=Props)
            RP2 = pd.DataFrame(RP2)

            # Keep Haversian canals not connected to border
            RP = pd.DataFrame(index=RP2['label'], columns=Props)
            for l in RP.index:
                A1 = RP1[RP1['label'] == l]['area'].values[0]
                A2 = RP2[RP2['label'] == l]['area'].values[0]
                if A1 == A2:
                    RP.loc[l] = RP1[RP1['label'] == l][Props].values[0]
            RP = RP.dropna()
            RP['Aspect Ratio (-)'] = RP[Props[1]] / RP[Props[2]]

            Data.loc[Index]['Haversian canals','Area (um2)'] = RP['area'].mean()*PS**2
            Data.loc[Index]['Haversian canals','Number (-)'] = len(RP)
            Data.loc[Index]['Haversian canals','L1 (um)'] = RP[Props[1]].mean()*PS
            Data.loc[Index]['Haversian canals','L2 (um)'] = RP[Props[2]].mean()*PS
            Data.loc[Index]['Haversian canals','Aspect Ratio (-)'] = RP['Aspect Ratio (-)'].mean()
            Data.loc[Index]['Haversian canals','Characteristic Length (um)'] = np.sqrt(RP['area'].mean()*PS**2)

            # Measure cement lines thickness
            MA, D = morphology.medial_axis(CL, return_distance=True)
            Data.loc[Index]['Cement lines','Thickness (um)'] = np.mean(D[MA])*PS

            # Compute relative densities
            HC = HC.sum() / HC.size * 100
            OS = OS.sum() / OS.size * 100
            CL = CL.sum() / CL.size * 100

            Data.loc[Index]['Haversian canals', 'Density (%)'] = HC
            Data.loc[Index]['Osteocytes', 'Density (%)'] = OS / (1- HC/100)
            Data.loc[Index]['Cement lines', 'Density (%)'] = CL / (1 - HC/100)

        # Update time
        Time.Update((iF+1)/len(Folders))

        # Save data
        Data.to_csv(ResDir / 'SegmentationData.csv')
        HCHC_Distribution.to_csv(ResDir / 'HCHC_Distances.csv')
        HCOS_Distribution.to_csv(ResDir / 'HCOS_Distances.csv')
        HCCL_Distribution.to_csv(ResDir / 'HCCL_Distances.csv')
        OSOS_Distribution.to_csv(ResDir / 'OSOS_Distances.csv')
        HC_Distribution.to_csv(ResDir / 'HC_Distances.csv')
        OS_Distribution.to_csv(ResDir / 'OS_Distances.csv')
        CL_Distribution.to_csv(ResDir / 'CL_Distances.csv')
        ImageQuality.to_csv(ResDir / 'ImageQuality.csv')
        OS_Colors.to_csv(ResDir / 'OS_Colors.csv')

    # Stop time tracking
    Time.Process(0)

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

