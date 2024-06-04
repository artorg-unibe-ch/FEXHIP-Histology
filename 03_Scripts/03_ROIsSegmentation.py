#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script performs automatic segmentation of blue stained
    cortical bone using U-net and random forest

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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
from pathlib import Path
from keras import utils, Model
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import feature, io, measure, morphology
from Utils import Time, Reference, StainNA, FeaturesExtraction

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

def Segmentation(Image:np.array, FeaturesExtractor:Model, RFc) -> np.array:

    # Record elapsed time
    Time.Process(1, 'Segment ROI')

    # Perform stain normalization
    Time.Update(1/5, 'Stain normalization')
    Norm = StainNA(Image, Reference.Mean, Reference.Std)

    # Extract features
    Time.Update(2/5, 'Extract Features')
    FE = FeaturesExtraction(FeaturesExtractor, Norm)
    Features = FE.reshape(-1, FE.shape[-1])

    # Get random forest prediction
    Time.Update(4/5, 'RF prediction')
    RFPred = RFc.predict(Features)
    RFPred = RFPred.reshape(Image.shape[:-1])

    Time.Process(0)

    return RFPred

def RemoveIsland(Mask:np.array, Threshold:int) -> np.array:

    """
    Remove island smaller than a given size
    """

    Regions = measure.label(Mask)
    Values, Counts = np.unique(Regions, return_counts=True)
    Cleaned = np.isin(Regions, Values[1:][Counts[1:] > Threshold])

    return Cleaned

def ReattributePixels(Cleaned:np.array, Mask:np.array, Values:list) -> np.array:

    """
    Modify pixel labels by the given values
    """

    Cleaned[(Cleaned == Values[1])*~Mask] = Values[0]
    Cleaned[(Cleaned == Values[1])* Mask] = Values[1]
    Cleaned[(Cleaned == Values[0])*~Mask] = Values[0]
    Cleaned[(Cleaned == Values[0])* Mask] = Values[1]

    return Cleaned

def CleanSegmentation(Pred:np.array) -> np.array:

    """
    Clean segmentation by connected component thresholding
    and thin cement lines by erosion 
    """

    # Get cement lines, Haversian canals, and osteocytes masks
    Clean = Pred.copy()
    OS = Clean == 2
    HC = Clean == 3
    CL = Clean == 4

    # Keep connected regions with more than 25 pixels
    CleanOS = RemoveIsland(OS, 25)

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanOS, [1,2])

    # Keep connected regions with more than 200 pixels
    CleanHC = RemoveIsland(HC, 200)

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanHC, [1,3])

    # Pad array to connect cement lines at the border
    Pad = np.pad(CL+OS, ((1,1),(1,1)), mode='constant', constant_values=True)

    # Keep connected regions with more than 300 pixels
    CleanCL = RemoveIsland(Pad, 300)

    # Thin cement lines by erosion
    CleanCL = morphology.isotropic_erosion(CleanCL, 1)
    CleanCL = CleanCL[1:-1,1:-1] * CL

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanCL, [1,4])

    return Clean

def SaveSegmentation(I:np.array, S:np.array, FigName:str) -> None:

    """
    Save segmentation mask
    """

    Categories = utils.to_categorical(S)

    # MaskName = FigName.parent / (FigName.name[:-4] + '_Mask.png')

    FigSize = np.array(I.shape[:-1])/100
    Figure, Axis = plt.subplots(1,1,figsize=FigSize, dpi=100)
    Axis.imshow(np.round(Categories[:,:,-3:] * 255).astype('int'))
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(FigName, dpi=100)
    plt.close(Figure)
    
    # Seg = np.zeros(S.shape + (4,))
    # Seg[:,:,:-1] = Categories[:,:,2:] * 255
    # BG = Categories[:,:,1].astype('bool')
    # Seg[:,:,-1][~BG] = 255
    # Seg[:,:,1][Seg[:,:,2] == 255] = 255

    # FigSize = np.array(I.shape[:-1])/100
    # Figure, Axis = plt.subplots(1,1,figsize=FigSize, dpi=100)
    # Axis.imshow(I)
    # Axis.imshow(Seg.astype('uint8'), alpha=0.5)
    # Axis.axis('off')
    # plt.subplots_adjust(0,0,1,1)
    # plt.savefig(FigName, dpi=100)
    # plt.close(Figure)

    return

def PlotOverlay(Image:np.array, Segmentation:np.array) -> None:

    # Plot overlay
    DPI=196
    S = Segmentation.shape
    Seg = np.zeros(S + (4,), 'uint8')
    Seg[Segmentation==2] = [255,   0,   0, 255]
    Seg[Segmentation==3] = [  0, 255,   0, 255]
    Seg[Segmentation==4] = [  0, 255, 255, 255]

    Figure, Axis = plt.subplots(1,1, figsize=(2*S[0]/DPI, 2*S[1]/DPI))
    Axis.imshow(Image)
    Axis.imshow(Seg.astype('uint8'), alpha=0.5)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1, wspace=0.05, hspace=0.05)
    plt.show(Figure)

    return

#%% Main
# Main part

def Main():

    # Set paths
    Dirs = SetDirectories('FEXHIP-Histology')
    ModelDir = Dirs['Results'] / '01_Training'
    DataDir = Dirs['Results'] / '02_ROIs'
    ResDir = Dirs['Results'] / '03_Segmentation'
    Path.mkdir(ResDir, exist_ok=True)

    # Build feature extractor
    Unet = load_model(ModelDir / 'UNet.hdf5')
    Outputs = [L.output for L in Unet.layers if 'conv' in L.name]
    FeaturesExtractor = Model(Unet.input, Outputs)

    # Load random forest classifier
    RFc = joblib.load(ModelDir / 'RandomForest.joblib')
    RFc.verbose = 0

    # List folders
    Folders = sorted([F for F in DataDir.iterdir() if Path.is_dir(F)])

    # Loop over every donor, every sample
    for Folder in Folders:
        Path.mkdir(ResDir / Folder.name, exist_ok=True)
        SegROIs = [F.name for F in Path.iterdir(ResDir / Folder.name)]

        # List and loop over ROI images
        Files = sorted([F for F in Folder.iterdir() if '_' in F.name])

        # Remove full images
        Files = [F for F in Files if not F.name.endswith('Zoom.png')]

        for File in Files:

            if File.name not in SegROIs:
                
                # Read and segment image
                Image = io.imread(File)
                PredRF = Segmentation(Image, FeaturesExtractor, RFc)

                # Clean cement lines segmentation
                Pred = CleanSegmentation(PredRF)
                
                # Save segmentation results
                FigName = ResDir / Folder.name / File.name
                SaveSegmentation(Image, Pred, FigName)

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

