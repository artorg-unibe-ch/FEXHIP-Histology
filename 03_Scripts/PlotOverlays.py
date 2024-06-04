#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script aims to plot overlays of ROI
    together with their segmentation

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: February 2024
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import argparse
import numpy as np
import pandas as pd
from keras import utils
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.morphology import disk
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from skimage import io, feature, morphology, measure

from Utils import Time, Unet_Probabilities, StainNA, Reference, CVAT

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

#%% Main
# Main part

def Main():

    # Set directories
    Dirs = SetDirectories('FEXHIP-Histology')
    ROIDir = Dirs['Results'] / '02_ROIs'
    MaskDir = Dirs['Results'] / '03_Segmentation'
    ResDir = Dirs['Results'] / '99_Overlays'
    Folders = sorted([F.name for F in ROIDir.iterdir()])

    # List ROIs
    for Folder in Folders:
        Files = [R.name for R in (ROIDir / Folder).iterdir() if not R.name.endswith('Zoom.png')]

        # Separate into diaphysis, neck sup and inf
        Diaphysis = [F for F in Files if F.startswith('Diaphysis')]
        NeckInf = [F for F in Files if 'Neck' in F and 'Inf' in F]
        NeckSup = [F for F in Files if 'Neck' in F and 'Sup' in F]

        # Separate further into left and right
        DiL = [F for F in Diaphysis if 'Left' in F]
        DiR = [F for F in Diaphysis if 'Right' in F]
        NIL = [F for F in NeckInf if 'Left' in F]
        NIR = [F for F in NeckInf if 'Right' in F]
        NSL = [F for F in NeckSup if 'Left' in F]
        NSR = [F for F in NeckSup if 'Right' in F]

        # Create folder and save plots
        Time.Process(1, Folder.replace('_',' '))
        os.makedirs(str(ResDir / Folder), exist_ok=True)
        Sites = [DiL, DiR, NIL, NIR, NSL, NSR]
        Names = ['Diaphysis_Left', 'Diaphysis_Right',
                'Neck_Inf_Left', 'Neck_Inf_Right',
                'Neck_Sup_Left', 'Neck_Sup_Right']
        DPI = 192
        for j, (Site, Name) in enumerate(zip(Sites, Names)):
            FName = str(ResDir / Folder / Name)
            if len(Site) >= 16:
                N = 16
                Samples = random.sample(Site,N)
            else:
                N = np.floor(np.sqrt(len(Site)))**2
                Samples = random.sample(Site,int(N))

            if N > 0:
                RC = int(np.sqrt(N))
                Figure, Axis = plt.subplots(RC,RC, figsize=(RC*S[0]/DPI, RC*S[1]/DPI))
                for i, Sample in enumerate(Samples):

                    R = i // RC
                    C = i - RC*R
                    
                    # Read image and mask
                    Image = io.imread(ROIDir / Folder / Sample)
                    Mask = io.imread(MaskDir / Folder / Sample)

                    # Plot overlay
                    S = Mask.shape
                    Seg = np.zeros(S)
                    Seg[:,:,:-1] = Mask[:,:,:-1]
                    BG = ~np.sum(Mask[:,:,:-1],axis=-1).astype('bool')
                    Seg[:,:,-1][~BG] = 255
                    Seg[:,:,1] = Mask[:,:,1] + Mask[:,:,2]

                    if RC > 1:
                        Axis[R,C].imshow(Image)
                        Axis[R,C].imshow(Seg.astype('uint8'), alpha=0.5)
                        Axis[R,C].axis('off')
                    else:
                        Axis.imshow(Image)
                        Axis.imshow(Seg.astype('uint8'), alpha=0.5)
                        Axis.axis('off')

                plt.subplots_adjust(0,0,1,1, wspace=0.05, hspace=0.05)
                plt.savefig(FName, dpi=DPI)
                plt.close(Figure)

            Time.Update((j+1)/len(Sites))
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
