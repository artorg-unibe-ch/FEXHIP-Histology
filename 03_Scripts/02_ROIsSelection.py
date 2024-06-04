#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform automatic selection of regions of
    interests (ROIs) to segment

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: October 2023
    """

#%% Imports
# Modules import

import PIL
import argparse
import numpy as np
from Utils import Time
from pathlib import Path
from patchify import patchify
import matplotlib.pyplot as plt
from skimage import io, measure


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

def SegmentBone(Image:np.array, Plot=False) -> np.array:

    """
    Segment bone structure based on RGB threshold
    :param Image: RGB numpy array dim r x c x 3
    :param Plot: Plot the results (bool)
    :return: Segmented bone image
    """
    
    # Mark areas where there is bone
    Filter1 = Image[:, :, 0] > 160
    Filter2 = Image[:, :, 1] > 160
    Filter3 = Image[:, :, 2] > 160
    Filter4 = Image[:, :, 2] < 130
    Bone = ~(Filter1 & Filter2 & Filter3 | Filter4)

    if Plot:
        Shape = np.array(Image.shape) / max(Image.shape) * 10
        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Image)
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

        Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]))
        Axis.imshow(Bone, cmap='binary')
        Axis.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.show()

    # Keep connected component
    Labels = measure.label(Bone)
    Areas = measure.regionprops_table(Labels, properties=['area'])['area']
    KArea = np.where(Areas > 1E6)[0]
    Bone = np.zeros(Labels.shape)
    for K in KArea:
        Bone += Labels == K + 1

    return Bone

def RandomROIs(Valid:np.array, Rand:list, Spacing:int) -> (np.array, list):

    """
    Select a region of interest at a random location for a given
    valid region defined in a numpy array
    """

    Y, X = np.where(Valid)
    RandInt = np.random.randint(len(X))
    Xc, Yc = X[RandInt], Y[RandInt]
    Y1, Y2 = Yc-Spacing+1, Yc+Spacing
    X1, X2 = Xc-Spacing+1, Xc+Spacing
    if Y1 < 0:
        Y1 = 0
    if X1 < 0:
        X1 = 0
    if Y2 >= Valid.shape[0]:
        Y2 = Valid.shape[0] - 1
    if X2 >= Valid.shape[1]:
        X2 = Valid.shape[1] - 1
    Valid[Y1:Y2, X1:X2] = False
    Rand.append((Xc, Yc))

    return Valid, Rand

def ExtractROIs(Image:np.array, Bone:np.array, N:int, ROIName:str) -> (list, list):

    """
    Extract regions of interest of cortical bone according to the parameters given as arguments for the Main function.
    According to Grimal et al (2011), cortical bone representative volume element should be around 1mm side length and
    presents a BV/TV of 88% at least. Therefore, a threshold of 0.88 is used to ensure that the selected ROI reaches
    this value.

    Grimal, Q., Raum, K., Gerisch, A., &#38; Laugier, P. (2011)
    A determination of the minimum sizes of representative volume elements
    for the prediction of cortical bone elastic properties
    Biomechanics and Modeling in Mechanobiology (6), 925-937
    https://doi.org/10.1007/s10237-010-0284-9

    :param Array: 3D numpy array (2D + RGB)
    :param Bone: 2D numpy array of segmented bone (bool)
    :param N: Number of ROIs to extract (int)
    :param Plot: Plot the results (bool)
    :return: ROIs
    """

    # Fixed parameters
    Threshold = 0.88                # BV/TV threshold
    Pixel_S = 1.046                 # Pixel spacing
    ROI_S = 500                     # ROI physical size

    # Set ROI pixel size
    ROISize = int(round(ROI_S / Pixel_S))

    # Patchify image and keep ROIs with BV/TV > Threshold
    Spacing = 8
    Step = (ROISize//Spacing, ROISize//Spacing)
    iStep = Step + (3,)
    Patches = patchify(Bone, patch_size=(ROISize, ROISize), step=Step)
    Valid = np.sum(Patches, axis=(2,3)) / (ROISize**2) > Threshold

    if np.sum(Valid) > 0:

        # Select random ROIs
        Rand = []
        Valid, Rand = RandomROIs(Valid, Rand, Spacing)
        while len(Rand) < N and np.sum(Valid) > 0:
            Valid, Rand = RandomROIs(Valid, Rand, Spacing)

        Patches = patchify(Image, patch_size=(ROISize, ROISize, 3), step=iStep)

        Xs = []
        Ys = []
        for i, (Rx, Ry) in enumerate(Rand):
            ROI = Patches[Ry, Rx][0].astype('uint8')
            Name = str(ROIName) + '_' + f'{i:02d}' + '.png'
            io.imsave(Name, ROI)
            X0 = Rx * Step[0]
            Y0 = Ry * Step[1]
            Xs.append([X0, X0 + ROISize])
            Ys.append([Y0, Y0 + ROISize])

        Xs = np.array(Xs)
        Ys = np.array(Ys)
       
    else:
        Xs, Ys = np.array([]), np.array([])
        print('\nCan\'t find ROI with BV/TV > ' + str(Threshold))

    return Xs, Ys

def PlotROIs(Bone:np.array, Image:np.array,
             ROIName:str, Xs:np.array, Ys:np.array) -> None:
    
    """
    Plot original image with selected ROIs in red contour
    Overlay a green area to show bone segmentation results
    """

    # Downsample arrays for plotting
    Factor = 12
    dArray = Image[::Factor, ::Factor]
    dBone = Bone[::Factor, ::Factor]
    Xs = Xs / Factor
    Ys = Ys / Factor

    # Bone segmentation
    dSeg = np.zeros(dBone.shape + (4, ), int)
    dSeg[:,:,1] = np.array(dBone * 255).astype(int)
    dSeg[:,:,-1] = np.array(dBone * 255).astype(int)

    # Plot full image with ROIs location and overlay bone segmentation
    Shape = np.array(Image.shape[:-1]) / 1000
    Figure, Axis = plt.subplots(1, 1, figsize=(Shape[1], Shape[0]), dpi=100)
    Axis.imshow(dArray)
    # Axis.imshow(dSeg, alpha=0.25)
    for i in range(len(Xs)):
        Axis.plot([Xs[i,0], Xs[i,1]], [Ys[i,0], Ys[i,0]], color=(1, 0, 0), linewidth=5)
        Axis.plot([Xs[i,1], Xs[i,1]], [Ys[i,0], Ys[i,1]], color=(1, 0, 0), linewidth=5)
        Axis.plot([Xs[i,1], Xs[i,0]], [Ys[i,1], Ys[i,1]], color=(1, 0, 0), linewidth=5)
        Axis.plot([Xs[i,0], Xs[i,0]], [Ys[i,1], Ys[i,0]], color=(1, 0, 0), linewidth=5)
    Axis.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(str(ROIName) + '.png', dpi=100)
    plt.close()

    return

#%% Main
# Main part

def Main(N=40):

    # Record time
    Time.Process(1, 'Select ROIs')

    # Remove max image pixels number
    PIL.Image.MAX_IMAGE_PIXELS = None

    # Set directories
    Dirs = SetDirectories('FEXHIP-Histology')
    DataDir = Dirs['Data']
    Folders = sorted([F for F in DataDir.iterdir() if F.name.startswith('Donor')])
    Folders = Folders[:-1]

    # Create results directory
    ResDir = Dirs['Results'] / '02_ROIs'
    Path.mkdir(ResDir, exist_ok=True)

    # Loop for each image for each folder
    for i, Folder in enumerate(Folders):

        Time.Update((i+1)/len(Folders), f'Folder n°{i:02d}')

        # Create ROI result directory
        ROIDir = ResDir / Folder.name
        Path.mkdir(ROIDir, exist_ok=True)
        ROIs = set([R.name[:-7] for R in ROIDir.iterdir() if not R.name.endswith('Zoom.png')])

        # Segment images by thresholding to define bone
        Files = sorted([F for F in Folder.iterdir() if 'Zoom' in F.name])
        for File in Files:

            if File.name[:-4] not in ROIs:

                Time.Update((i+1)/len(Folders), 'Read image')
                Image = io.imread(File)

                Time.Update((i+1)/len(Folders), 'Segment bone')
                Bone = SegmentBone(Image)

                # Based on bone pixels coordinates
                # Select N ROIs with BV/TV > 0.88
                Time.Update((i+1)/len(Folders), 'Extract ROIs')
                ROIName = ROIDir / File.name[:-4]
                Xs, Ys = ExtractROIs(Image, Bone, N, ROIName)

                Time.Update((i+1)/len(Folders), 'Plot ROIs')
                PlotROIs(Bone, Image, ROIName, Xs, Ys)

    Time.Process(0)

    return

#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)
    Parser.add_argument('-N', '--Number', help='Number of ROIs', default=40, type=int)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments.Number)
# %%
