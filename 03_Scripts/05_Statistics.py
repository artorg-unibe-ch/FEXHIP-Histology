#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform statistical analysis of
    automatic segmentation results

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2023
    """

#%% Imports
# Modules import

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from Utils import Time, CVAT #type:ignore
import matplotlib.pyplot as plt
from matplotlib.cm import winter, jet
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from scipy.stats.distributions import norm, t, chi2, f
from scipy.stats import pearsonr, shapiro, kstest, ttest_rel, ttest_ind


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

def DataStats(Data:pd.DataFrame) -> None:

    """
    Print main statistics of the data frame
    """

    print('\nData Statistics:')
    DF = Data.groupby(level=2)
    NDiaDonors = len(DF.get_group('Diaphysis').groupby(level=[0]))
    print('Number of diaphysis donors: ', NDiaDonors)
    NDiaphysis = len(DF.get_group('Diaphysis').groupby(level=[0,1]))
    print('Number of diaphyses: ', NDiaphysis)
    NSupDonors = len(DF.get_group('Neck Superior').groupby(level=[0]))
    print('Number of superior neck donors: ', NSupDonors)
    NSuperior = len(DF.get_group('Neck Superior').groupby(level=[0,1]))
    print('Number of superior necks: ', NSuperior)
    NInfDonors = len(DF.get_group('Neck Inferior').groupby(level=[0]))
    print('Number of inferior neck donors: ', NInfDonors)
    NInferior = len(DF.get_group('Neck Inferior').groupby(level=[0,1]))
    print('Number of inferior necks: ', NInferior)

    return
    
def ShowROI(Idx:tuple, Dirs:dict, FigSave='') -> None:

    """
    Plot a specific ROI given data frame index
    """

    Donor = f'Donor_{Idx[0]:02d}'
    Site = Idx[2].split()
    if len(Site) == 1:
        ROI = Site[0] + '_' + Idx[1] + '_Zoom_' + f'{Idx[3]-1:02d}.png'
    else:
        ROI = Site[0] + '_' + Idx[1] + '_' + Site[1][:3] + '_Zoom_' + f'{Idx[3]-1:02d}.png'
    FName = Dirs['Results'] / '02_ROIs' / Donor / ROI

    Image = io.imread(FName)
    Size = Image.shape
    DPI = 96
    FigSize = np.array(Size) / DPI

    Figure, Axis = plt.subplots(1,1, figsize=(FigSize[0], FigSize[1]), dpi=DPI)
    Axis.imshow(Image)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    if len(FigSave) > 0:
        plt.savefig(FigSave, dpi=DPI)
    plt.show(Figure)

    return

def ShowQuality(Data:pd.DataFrame, Features:list, Dirs:dict, NValues=3, FigSave='') -> None:

    """
    Plot ROIs of different quality levels according to a specific feature
    """

    for iF, F in enumerate(Features):

        Values = np.linspace(Data[F].min(), Data[F].max(), NValues)

        for i, Value in enumerate(Values):
            
            Idx = (Data[F] - Value).abs().idxmin()

            Donor = f'Donor_{Idx[0]:02d}'
            Site = Idx[2].split()
            if len(Site) == 1:
                ROI = Site[0] + '_' + Idx[1] + '_Zoom_' + f'{Idx[3]-1:02d}.png'
            else:
                ROI = Site[0] + '_' + Idx[1] + '_' + Site[1][:3] + '_Zoom_' + f'{Idx[3]-1:02d}.png'
            FName = Dirs['Results'] / '02_ROIs' / Donor / ROI

            Image = io.imread(FName)

            if iF == 0 and i==0:
                Size = Image.shape
                DPI = 96
                FigSize = np.array(Size) / DPI

                Figure, Axis = plt.subplots(len(Features),len(Values), figsize=(FigSize[0]*len(Values), FigSize[1]*len(Features)), dpi=DPI)
        
            if len(Features) > 1 and len(Values) > 1:
                Axis[iF,i].imshow(Image)
                Axis[iF,i].axis('off')

            else:
                Axis[iF+i].imshow(Image)
                Axis[iF+i].axis('off')

    plt.subplots_adjust(0,0,1,1,0.05, 0.05)
    if len(FigSave) > 0:
        plt.savefig(FigSave, dpi=DPI)
    plt.show(Figure)

    return

def Scatter3D(X:np.array, Y:np.array, Z:np.array, Colors:np.array) -> None:

    """
    Perform a 3D scatter plot of X,Y,Z data for given colors
    """

    Figure = plt.figure(figsize=(5.5, 4))
    Axis = Figure.add_subplot(111, projection='3d')
    Axis.scatter(X, Y, Z, c=Colors)

    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axis.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    
    # make the panes transparent
    Axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    
    # modify ticks
    MinX, MaxX = 0, 255
    MinY, MaxY = 0, 255
    MinZ, MaxZ = 0, 255
    Axis.set_xticks([MinX, MaxX])
    Axis.set_yticks([MinY, MaxY])
    Axis.set_zticks([MinZ, MaxZ])
    Axis.xaxis.set_ticklabels([MinX, MaxX])
    Axis.yaxis.set_ticklabels([MinY, MaxY])
    Axis.zaxis.set_ticklabels([MinZ, MaxZ])

    Axis.set_xlabel('R')
    Axis.set_ylabel('G')
    Axis.set_zlabel('B')

    plt.show()

    return

def Histogram(Array:np.array, FigName:str, Labels=[], Bins=20) -> None:

    """
    Plot data histogram along with kernel density and
    corresponding normal distribution to assess data
    normality
    """

    # Compute data values
    X = pd.DataFrame(Array, dtype='float')
    SortedValues = np.sort(X.T.values)[0]
    N = len(X)
    X_Bar = X.mean()
    S_X = np.std(X, ddof=1)

    # Figure plotting
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)

    # Histogram
    Histogram, Edges = np.histogram(X, bins=Bins, density=True)
    Width = 0.9 * (Edges[1] - Edges[0])
    Center = (Edges[:-1] + Edges[1:]) / 2
    Axes.bar(Center, Histogram, align='center', width=Width,
                edgecolor=(0,0,0), color=(1, 1, 1, 0), label='Histogram')

    # Density distribution
    KernelEstimator = np.zeros(N)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
    KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
    for Value in SortedValues:
        KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
    KernelEstimator = KernelEstimator / N

    Axes.plot(SortedValues, KernelEstimator, color=(1,0,0), label='Kernel density')

    # Corresponding normal distribution
    TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
    Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--',
                color=(0,0,1), label='Normal distribution')
    
    if len(Labels) > 0:
        plt.xlabel(Labels[0])
        plt.ylabel(Labels[1])

    plt.legend(loc='best')
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return

def QQPlot(Array:np.array, FigName:str, Alpha_CI=0.95) -> float:

    """
    Show quantile-quantile plot
    Add Shapiro-wilk test p-value to estimate
    data normality distribution assumption
    Based on: https://www.tjmahr.com/quantile-quantile-plots-from-scratch/
    Itself based on Fox book: Fox, J. (2015)
    Applied Regression Analysis and Generalized Linear Models.
    Sage Publications, Thousand Oaks, California.
    """

    # Shapiro-Wilk test for normality
    W, p = shapiro(Array)

    # Data analysis
    DataValues = pd.DataFrame(Array, dtype='float')
    N = len(DataValues)
    X_Bar = np.mean(DataValues, axis=0)
    S_X = np.std(DataValues,ddof=1)

    # Sort data to get the rank
    Data_Sorted = DataValues.sort_values(0)
    Data_Sorted = np.array(Data_Sorted).ravel()

    # Compute quantiles
    EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
    TheoreticalQuantiles = norm.ppf(EmpiricalQuantiles, X_Bar, S_X)
    ZQuantiles = norm.ppf(EmpiricalQuantiles,0,1)

    # Compute data variance
    DataIQR = np.quantile(DataValues, 0.75) - np.quantile(DataValues, 0.25)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    Variance = DataIQR / NormalIQR
    Z_Space = np.linspace(min(ZQuantiles), max(ZQuantiles), 100)
    Variance_Line = Z_Space * Variance + np.median(DataValues)

    # Compute alpha confidence interval (CI)
    Z_SE = np.sqrt(norm.cdf(Z_Space) * (1 - norm.cdf(Z_Space)) / N) / norm.pdf(Z_Space)
    Data_SE = Z_SE * Variance
    Z_CI_Quantile = norm.ppf(np.array([(1 - Alpha_CI) / 2]), 0, 1)

    # Create point in the data space
    Data_Space = np.linspace(min(TheoreticalQuantiles), max(TheoreticalQuantiles), 100)

    # QQPlot
    BorderSpace = max(0.05*abs(Data_Sorted.min()), 0.05*abs(Data_Sorted.max()))
    Y_Min = Data_Sorted.min() - BorderSpace
    Y_Max = Data_Sorted.max() + BorderSpace

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.plot(Data_Space, Variance_Line, linestyle='--', color=(1, 0, 0), label='Variance :' + str(format(np.round(Variance, 2),'.2f')))
    Axes.plot(Data_Space, Variance_Line + Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1), label=str(int(100*Alpha_CI)) + '% CI')
    Axes.plot(Data_Space, Variance_Line - Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1))
    Axes.plot(TheoreticalQuantiles, Data_Sorted, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 0))
    Axes.text(0.05,0.9,'Shapiro-Wilk p-value: ' + str(round(p,3)),transform=Axes.transAxes)
    plt.xlabel('Theoretical quantiles (-)')
    plt.ylabel('Empirical quantiles (-)')
    plt.ylim([Y_Min, Y_Max])
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size':10})
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return p

def CDFPlot(Array:np.array, FigName:str, Xlabel:str) -> float:

    """
    Plot Empirical cumulative distribution function of
    given array and theorical cumulative distribution
    function of normal distribution. Adds Kolmogorov-Smirnoff
    test p-value to assess data normality ditribution assumption
    """

    # Kolmogorov-Smirnoff test for normality
    KS, p = kstest(Array)

    # Data analysis
    DataValues = pd.DataFrame(Array, dtype='float')
    N = len(DataValues)
    X_Bar = np.mean(DataValues, axis=0)
    S_X = np.std(DataValues,ddof=1)

    # Sort data to get the rank
    Data_Sorted = DataValues.sort_values(0)
    Data_Sorted = np.array(Data_Sorted).ravel()

    # Compute quantiles
    EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
    Z = (Data_Sorted - X_Bar) / S_X
    TheoreticalQuantiles = norm.cdf(Z)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.plot(Data_Sorted,EmpiricalQuantiles, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 1), label='Data Distribution')
    Axes.plot(Data_Sorted,TheoreticalQuantiles, linestyle='--', color=(1, 0, 0), label='Normal Distribution')
    Axes.text(0.05,0.9,'Kolmogorov-Smirnoff p-value: ' + str(round(p,3)),transform=Axes.transAxes)
    plt.xlabel(Xlabel)
    plt.ylabel('Quantile (-)')
    plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.15), prop={'size':10})
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return p

def PlotOLS(X:np.array, Y:np.array, Labels=None, Alpha=0.95, FName=None) -> None:
    
    """
    Plot linear regression between to variables X and Y


    Parameters
    ----------
    X: Independent variable
    Y: Dependent variable
    Labels: Labels for the different axes/variables (X and Y)
    Alpha: Conficence level
    FName: Figure name (to save it)

    Returns
    -------
    None
    """

    if Labels == None:
        Labels = ['X', 'Y']
    
    # Perform linear regression
    Xm = np.matrix([np.ones(len(X)), X]).T
    Ym = np.matrix(Y).T
    Intercept, Slope = np.linalg.inv(Xm.T * Xm) * Xm.T * Ym
    Intercept = np.array(Intercept)[0,0]
    Slope = np.array(Slope)[0,0]

    # Build arrays and matrices
    Y_Obs = Y
    Y_Fit = X * Slope + Intercept
    N = len(Y)
    X = np.matrix(X)

    # Sort X values and Y accordingly
    Sort = np.argsort(np.array(Xm[:,1]).reshape(len(Xm)))
    X_Obs = np.sort(np.array(Xm[:,1]).reshape(len(Xm)))
    Y_Fit = Y_Fit[Sort]
    Y_Obs = Y_Obs[Sort]

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / (N - 2))
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS
    R2adj = 1 - RSS/TSS * (N-1)/(N-Xm.shape[1]+1-1)

    ## Compute variance-covariance matrix
    C = np.linalg.inv(Xm.T * Xm)

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(Xm * C * Xm.T)))
    t_Alpha = t.interval(Alpha, N - Xm.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0[Sort]
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0[Sort]

    # Plots
    DPI = 96
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI)
    Axes.plot(X_Obs, Y_Obs, linestyle='none', marker='o', color=(0,0,1), fillstyle='none')
    Axes.plot(X_Obs, Y_Fit, color=(1,0,0))
    Axes.fill_between(X_Obs, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)

    # Add annotations
    if Slope > 0:

        # Number of observations
        YPos = 0.925
        Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos -= 0.075
        Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos -= 0.075
        Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
        
        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.025
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] *  np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin], Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
        YPos += 0.075

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin], Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')

    elif Slope < 0:

        # Number of observations
        YPos = 0.025
        Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos += 0.075
        Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos += 0.075
        Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.925
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin],Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
        YPos -= 0.075

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin],Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axes.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
    
    Axes.set_xlabel(Labels[0])
    Axes.set_ylabel(Labels[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if FName:
        plt.savefig(FName, dpi=196)
    plt.show(Figure)

    return

# def PlotMixedLM(Data:pd.DataFrame, LME:smf.mixedlm, Model:smf.mixedlm, FigName=None,    #type:ignore
#                 Xlabel='X', Ylabel='Y', Alpha_CI=0.95) -> None:
    
#     """
#     Function used to plot mixed linear model results
#     Plotting based on: https://www.azandisresearch.com/2022/12/31/visualize-mixed-effect-regressions-in-r-with-ggplot2/
#     As bootstrap is expensive for CI band computation, compute
#     CI bands based on FOX 2017

#     Only implemented for 2 levels LME with nested random intercepts
#     """

#     # Compute conditional residuals
#     Data['CR'] = LME.params[0] + Data['X']*LME.params[1] + LME.resid

#     # Create X values for confidence interval lines
#     Min = Data['X'].min()
#     Max = Data['X'].max()
#     Range = np.linspace(Min, Max, len(Data))

#     # Get corresponding fitted values and CI interval
#     Y_Fit = LME.params[0] + Range * LME.params[1]
#     Alpha = t.interval(Alpha_CI, len(Data) - len(LME.fe_params) - 1)

#     # Residual sum of squares
#     RSS = np.sum(LME.resid ** 2)
#     TSS = np.sum((Data['Y'] - Data['Y'].mean()) ** 2)
#     RegSS = TSS - RSS
#     R2 = RegSS / TSS

#     # Standard error of the estimate
#     SE = np.sqrt(RSS / LME.df_resid)

#     # Compute corresponding CI lines
#     C = np.matrix(LME.normalized_cov_params)
#     X = np.matrix([np.ones(len(Data)),np.linspace(Min, Max, len(Data))]).T
    
#     if C.shape[0] > len(LME.fe_params):
#         C = C[:len(LME.fe_params),:len(LME.fe_params)]

#     B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))

#     CI_Line_u = Y_Fit + Alpha[0] * SE * B_0
#     CI_Line_o = Y_Fit + Alpha[1] * SE * B_0

#     # Different markers for different anatomical sites
#     Markers = ['o','x','^']

#     # Plot and save results
#     Figure, Axis = plt.subplots(1,1)
#     Axis.fill_between(Range, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)
#     Sites = []
#     for i, (S, Df) in enumerate(Data.groupby(by='Site')):
#         Sites.append(S)
#         Colors, Donors = [], []
#         for Donor, df in Df.groupby(by='Donor'):
#             Color = winter(Donor / max(Data['Donor'].values))
#             Axis.plot(df['X'], df['Y'], c=Color, linestyle='none',
#                         marker=Markers[i], fillstyle='none')
#             Colors.append(Color)
#             Donors.append(Donor)

#     Axis.plot(Range, Y_Fit, color=(1,0,0))

#     # Create legend
#     L1 = []
#     for M in Markers:
#         P, = plt.plot([],color=(0,0,0), marker=M, linestyle='none', fillstyle='none')
#         L1.append(P)

#     Axis.set_ylabel(Ylabel)
#     Axis.set_xlabel(Xlabel)
#     plt.legend(L1, Sites, loc='upper center', bbox_to_anchor=(0.5,1.1),ncol=3)
    
#     # Add annotations
#     Intercept = LME.params[0]
#     Slope = LME.params[1]
#     N = len(Data)
#     # if Slope > 0:

#     # Number of observations
#     YPos = 0.925
#     Axis.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

#     # Pearson's correlation coefficient
#     YPos -= 0.075
#     Axis.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

#     # Standard error of the estimate
#     YPos -= 0.075
#     Axis.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
    
#     # Intercept coeffecient and corresponding confidence interval
#     YPos = 0.025
#     Round = 4 - str(Intercept).find('.')
#     rIntercept = np.round(Intercept, Round)
#     CIMargin = Alpha[1] *  np.sqrt(RSS / (N - 2) * C[0,0])
#     CI = np.round([Intercept - CIMargin, Intercept + CIMargin], Round)
#     if Round <= 0:
#         rIntercept = int(rIntercept)
#         CI = [int(v) for v in CI]
#     Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
#     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
#     YPos += 0.075

#     # Slope coeffecient and corresponding confidence interval
#     Round = 3 - str(Slope).find('.')
#     rSlope = np.round(Slope, Round)
#     CIMargin = Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
#     CI = np.round([Slope - CIMargin, Slope + CIMargin], Round)
#     if Round <= 0:
#         rSlope = int(rSlope)
#         CI = [int(v) for v in CI]
#     Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
#     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')

#     # Group variability (BLUP)
#     RE = Fit.random_effects

#     # Multiply each BLUP by the random effects design matrix for one group
#     # Only take first value as only random intercept
#     REx = [np.dot(Model.exog_re_li[j][0], RE[k]) for (j, k) in enumerate(Model.group_labels)]

#     # Add annotation
#     YPos += 0.075
#     Std = np.std(REx, ddof=1)
#     Round = 3 - str(Std).find('.')
#     rStd = np.round(Std, Round)
#     Text = r'Donor $\sigma^2$ : ' + str(rStd) + '$^2$'
#     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')



#     # elif Slope < 0:

#     #     # Number of observations
#     #     YPos = 0.025
#     #     Axis.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

#     #     # Pearson's correlation coefficient
#     #     YPos += 0.075
#     #     Axis.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

#     #     # Standard error of the estimate
#     #     YPos += 0.075
#     #     Axis.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

#     #     # Intercept coeffecient and corresponding confidence interval
#     #     YPos = 0.925
#     #     Round = 3 - str(Intercept).find('.')
#     #     rIntercept = np.round(Intercept, Round)
#     #     CIMargin = Alpha[1] * np.sqrt(RSS / (N - 2) * C[0,0])
#     #     CI = np.round([Intercept - CIMargin, Intercept + CIMargin],Round)
#     #     if Round <= 0:
#     #         rIntercept = int(rIntercept)
#     #         CI = [int(v) for v in CI]
#     #     Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
#     #     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
#     #     YPos -= 0.075

#     #     # Slope coeffecient and corresponding confidence interval
#     #     Round = 3 - str(Slope).find('.')
#     #     rSlope = np.round(Slope, Round)
#     #     CIMargin = Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
#     #     CI = np.round([Slope - CIMargin, Slope + CIMargin],Round)
#     #     if Round <= 0:
#     #         rSlope = int(rSlope)
#     #         CI = [int(v) for v in CI]
#     #     Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
#     #     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
    
#     #     # Group variability (BLUP)
#     #     RE = Fit.random_effects

#     #     # Multiply each BLUP by the random effects design matrix for one group
#     #     # Only take first value as only random intercept
#     #     REx = [np.dot(LME.exog_re_li[j][0], RE[k]) for (j, k) in enumerate(LME.group_labels)]

#     #     # Add annotation
#     #     YPos -= 0.075
#     #     Std = np.std(REx, ddof=1)
#     #     Round = 3 - str(Std).find('.')
#     #     rStd = np.round(Std, Round)
#     #     Text = r'Donor $\sigma^2$ : ' + str(rStd) + '$^2$'
#     #     Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')

#     plt.subplots_adjust(left=0.15, bottom=0.15)

#     if FName:
#         plt.savefig(FName, dpi=196)
#     plt.show(Figure)

#     return

def PlotOLSDonors(Data:pd.DataFrame, Labels=None, Alpha=0.95, FName=None) -> None:
    
    """
    Plot linear regression between to variables X and Y


    Parameters
    ----------
    X: Independent variable
    Y: Dependent variable
    Labels: Labels for the different axes/variables (X and Y)
    Alpha: Conficence level
    FName: Figure name (to save it)

    Returns
    -------
    None
    """

    if Labels == None:
        Labels = ['X', 'Y']

    X = Data['X'].values
    Y = Data['Y'].values
    
    # Perform linear regression
    Xm = np.matrix([np.ones(len(X)), X]).T
    Ym = np.matrix(Y).T
    Intercept, Slope = np.linalg.inv(Xm.T * Xm) * Xm.T * Ym
    Intercept = np.array(Intercept)[0,0]
    Slope = np.array(Slope)[0,0]

    # Build arrays and matrices
    Y_Obs = Y
    Y_Fit = X * Slope + Intercept
    N = len(Y)
    X = np.matrix(X)

    # Sort X values and Y accordingly
    Sort = np.argsort(np.array(Xm[:,1]).reshape(len(Xm)))
    X_Obs = np.sort(np.array(Xm[:,1]).reshape(len(Xm)))
    Y_Fit = Y_Fit[Sort]
    Y_Obs = Y_Obs[Sort]

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / (N - 2))
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS
    R2adj = 1 - RSS/TSS * (N-1)/(N-Xm.shape[1]+1-1)

    ## Compute variance-covariance matrix
    C = np.linalg.inv(Xm.T * Xm)

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(Xm * C * Xm.T)))
    t_Alpha = t.interval(Alpha, N - Xm.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0[Sort]
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0[Sort]

    # Different markers for different anatomical sites
    Markers = ['o','x','^']

    # Store parameters
    Params = pd.DataFrame(columns=['Parameter','CI Low','CI High'],
                          index=['Intercept','Slope','R2','SE'])
    Params.loc['R2'] = [R2, 0, 0]
    Params.loc['SE'] = [SE, 0, 0]

    # Plot and save results
    Figure, Axis = plt.subplots(1,1)
    Axis.fill_between(X_Obs, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)
    Sites = []
    for i, (S, Df) in enumerate(Data.groupby(by='Site')):
        Sites.append(S)
        Colors, Donors = [], []
        for Donor, df in Df.groupby(by='Donor'):
            Color = winter(Donor / max(Data['Donor'].values))
            Axis.plot(df['X'], df['Y'], c=Color, linestyle='none',
                        marker=Markers[i], fillstyle='none')
            Colors.append(Color)
            Donors.append(Donor)

    Axis.plot(X_Obs, Y_Fit, color=(1,0,0))

    # Create legend
    L1 = []
    for M in Markers:
        P, = plt.plot([],color=(0,0,0), marker=M, linestyle='none', fillstyle='none')
        L1.append(P)

    Axis.set_ylabel(Labels[1])
    Axis.set_xlabel(Labels[0])
    plt.legend(L1, Sites, loc='upper center', bbox_to_anchor=(0.5,1.1),ncol=3)
    
    # Add annotations
    if Slope > 0:

        # Number of observations
        YPos = 0.925
        Axis.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos -= 0.075
        Axis.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos -= 0.075
        Axis.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.925
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] *  np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin], Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axis.annotate(Text, xy=(0.2, YPos), xycoords='axes fraction')
        YPos -= 0.075

        Params.loc['Intercept'] = [Intercept, Intercept-CIMargin, Intercept+CIMargin]

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin], Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axis.annotate(Text, xy=(0.2, YPos), xycoords='axes fraction')

        Params.loc['Slope'] = [Slope, Slope-CIMargin, Slope+CIMargin]

    elif Slope < 0:

        # Number of observations
        YPos = 0.025
        Axis.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')

        # Pearson's correlation coefficient
        YPos += 0.075
        Axis.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Standard error of the estimate
        YPos += 0.075
        Axis.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')

        # Intercept coeffecient and corresponding confidence interval
        YPos = 0.925
        Round = 3 - str(Intercept).find('.')
        rIntercept = np.round(Intercept, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[0,0])
        CI = np.round([Intercept - CIMargin, Intercept + CIMargin],Round)
        if Round <= 0:
            rIntercept = int(rIntercept)
            CI = [int(v) for v in CI]
        Text = r'Intercept : ' + str(rIntercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
        YPos -= 0.075

        Params.loc['Intercept'] = [Intercept, Intercept-CIMargin, Intercept+CIMargin]

        # Slope coeffecient and corresponding confidence interval
        Round = 3 - str(Slope).find('.')
        rSlope = np.round(Slope, Round)
        CIMargin = t_Alpha[1] * np.sqrt(RSS / (N - 2) * C[1,1])
        CI = np.round([Slope - CIMargin, Slope + CIMargin],Round)
        if Round <= 0:
            rSlope = int(rSlope)
            CI = [int(v) for v in CI]
        Text = r'Slope : ' + str(rSlope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
        Axis.annotate(Text, xy=(0.5, YPos), xycoords='axes fraction')
    
        Params.loc['Slope'] = [Slope, Slope-CIMargin, Slope+CIMargin]

    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if FName:
        plt.savefig(FName, dpi=196)
    plt.show(Figure)

    return Params

def PrintMorphoStats(Data:pd.DataFrame) -> None:

    """
    Print morphological statistics
    """

    Data[('Haversian canals','Diameter (um)')] = 2*np.sqrt(Data[('Haversian canals','Area (um2)')] / np.pi)

    Variables = [('Haversian canals','Density (%)'),
                 ('Haversian canals','Diameter (um)'),
                 ('Haversian canals','Characteristic Length (um)'),
                 ('Osteocytes','Density (%)'),
                 ('Osteocytes','L1 (um)'),
                 ('Osteocytes','L2 (um)'),
                 ('Osteocytes','Characteristic Length (um)')]

    for Site, df in Data.groupby('Site'):
        print('\n')
        print(Site)

        for V in Variables:
            print(V[0] + ' ' + V[1])
            Text = f'{round(df[V].mean(),3):.3f}' + u" \u00B1 " + f'{round(df[V].std(),3):.3f}'
            print(Text)

    return

def BoxPlot(ArraysList:list, Labels=['', 'Y'], SetsLabels=None,
            Vertical=True, FigName=None, YLim=[], Ttest=None) -> None:
    
    """
    Save boxplot of a list of arrays
    Used for assessment on random effects and residuals
    """

    if Vertical == True:
        Width = 2.5 + len(ArraysList)
        Figure, Axis = plt.subplots(1,1, dpi=100, figsize=(Width,4.5))
    else:
        Height = len(ArraysList) - 0.5
        Figure, Axis = plt.subplots(1,1, dpi=100, figsize=(6.5,Height))

    for i, Array in enumerate(ArraysList):

        # Create random positions
        Array = np.sort(Array)
        Norm = norm.pdf(np.linspace(-3,3,len(Array)), scale=1.5)
        Norm = Norm / max(Norm)
        RandPos = np.random.normal(0,0.03,len(Array)) * Norm + i

        if Vertical == True:
            Axis.plot(RandPos - RandPos.mean() + i, Array, linestyle='none',
                        marker='o',fillstyle='none', color=(1,0,0), ms=5)
        else:
            Axis.plot(Array, RandPos - RandPos.mean() + i, linestyle='none',
                        marker='o',fillstyle='none', color=(1,0,0), ms=5)
            
        Axis.boxplot(Array, vert=Vertical, widths=0.35,
                    showmeans=True,meanline=True,
                    showfliers=False, positions=[i],
                    capprops=dict(color=(0,0,0)),
                    boxprops=dict(color=(0,0,0)),
                    whiskerprops=dict(color=(0,0,0),linestyle='--'),
                    medianprops=dict(color=(0,1,0)),
                    meanprops=dict(color=(0,0,1)))

    if Ttest:
        for i, A in enumerate(ArraysList[:-1]):

            # Perform t-test
            if Ttest == 'Rel':
                T_Tests = ttest_rel(np.array(ArraysList[i+1],float), np.array(A,float))
            else:
                T_Tests = ttest_ind(np.array(ArraysList[i+1],float), np.array(A,float))
            YLine = 1.05 * max(A.max(), ArraysList[i+1].max())
            Plot = Axis.plot([i+0.05, i+0.95], [YLine, YLine], color=(0,0,0), marker='|',linewidth=0.5)
            MarkerSize = Plot[0].get_markersize()
            
            # Mark significance level
            if T_Tests[1] < 0.001:
                Text = '***'
            elif T_Tests[1] < 0.01:
                Text = '**' 
            elif T_Tests[1] < 0.05:
                Text = '*'
            else:
                Text = 'n.s.'
            Axis.annotate(Text, xy=[i+0.5, YLine], ha='center',
                          xytext=(0, -1.5*MarkerSize), textcoords='offset points',)

            # Write confidence interveal
            CIl = round(T_Tests.confidence_interval()[0],1)
            CIu = round(T_Tests.confidence_interval()[1],1)
            Text = 'CI (' + str(CIl) + ',' + str(CIu) + ')'
            Axis.annotate(Text, xy=[i+0.5, YLine], ha='center',
                          xytext=(0, 1.2*MarkerSize), textcoords='offset points',)
            if i == 0:
                Max = YLine*1.05
            else:
                Max = max([Max, YLine*1.05])
            Axis.set_ylim([0.95*min([min(A)for A in ArraysList]), Max])
    
    Axis.plot([],linestyle='none',marker='o',fillstyle='none', color=(1,0,0), label='Data')
    Axis.plot([],color=(0,0,1), label='Mean', linestyle='--')
    Axis.plot([],color=(0,1,0), label='Median')
    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])

    if SetsLabels and Vertical==True:
        Axis.set_xticks(np.arange(len(SetsLabels)))
        Axis.set_xticklabels(SetsLabels, rotation=0)
    elif SetsLabels and Vertical==False:
        Axis.set_yticks(np.arange(len(SetsLabels)))
        Axis.set_yticklabels(SetsLabels, rotation=0)
    else:
        Axis.set_xticks([])

    if len(YLim) == 2:
        Axis.set_ylim(YLim)
    
    if Vertical == True:
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.125))
        plt.subplots_adjust(left=0.25, right=0.75)
    else:
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
        plt.subplots_adjust(left=0.25, right=0.75,top=0.8)
    
    if FigName:
        plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=196)
    plt.show(Figure)

    return

def PlotDistribution(Data:pd.DataFrame, Labels=['',''], XLim=[], FName=None) -> list:

    """
    Plot mean distances distributions +- their standard deviation
    """

    Average = Data.mean().values * 100
    EnvelopeUp = Average + Data.std().values * 100
    EnvelopeLow = Average - Data.std().values * 100
    D50Curve = int(Data.columns[np.argmin(np.abs(Average - 50))])
    D95Curve = int(Data.columns[np.argmin(np.abs(Average - 95))])

    # Compute D50, D95  and their standard deviation
    D50 = np.array((Data - 0.5).abs().idxmin(axis=1),float)
    D95 = np.array((Data - 0.95).abs().idxmin(axis=1),float)
    T50 = str(round(D50.mean())) + ' $\pm$ ' + str(round(D50.std()))
    T95 = str(round(D95.mean())) + ' $\pm$ ' + str(round(D95.std()))

    Figure, Axis = plt.subplots(1,1, dpi=192)
    Axis.plot([D50Curve, D50Curve], [5, 50], color=(0,0,0), linewidth=1)
    Axis.plot([D95Curve, D95Curve], [55, 95], color=(0,0,0), linewidth=1)
    Axis.plot(np.array(Data.columns,'int'), Average, color=(1,0,0))
    Axis.fill_between(np.array(Data.columns,'int'),
                    EnvelopeUp, EnvelopeLow,
                    color=(0,0,0,0.1))
    Axis.annotate('D$_{50}$ = ' + T50, xy=(D50Curve+10,10))
    Axis.annotate('D$_{95}$ = ' + T95, xy=(D95Curve+10,60))
    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])

    if len(XLim) == 2:
        Axis.set_xlim(XLim)
    
    if FName:
        plt.savefig(FName)
    plt.show(Figure)

    return [np.array([D50.mean(), D50.std()]), np.array([D95.mean(), D95.std()])]

def ANOVA(DataTest:pd.DataFrame, Alpha=0.05, Labels=['', 'Y'], YLim=[], FigName='') -> pd.DataFrame:

    # Get columns
    Columns = DataTest.columns
    ANOVA_Table = pd.DataFrame(index=['Treatment','Error'],
                               columns=['SS','DF','MS'])

    # Get values (remove nans)
    isNA = [DataTest[C].isna() for C in Columns]

    # Compute means
    y = [DataTest[C][~isNA[i]].values for i, C in enumerate(Columns)]
    y_i = [np.mean(i) for i in y]
    y_ = np.mean(y_i)

    # Compute sum of square
    SST = 0
    for i in range(len(Columns)):
        SST += len(y[i]) * (y_i[i] - y_)**2

    SS = sum([sum((i - y_)**2) for i in y])
    SSE = SS - SST

    ANOVA_Table.loc['Treatment','SS'] = SST
    ANOVA_Table.loc['Error','SS'] = SSE

    # Degrees of freedom
    DFT = len(Columns) - 1
    DFE = sum([len(i) for i in y]) - len(Columns)

    ANOVA_Table.loc['Treatment','DF'] = DFT
    ANOVA_Table.loc['Error','DF'] = DFE

    # Compute F-statistic
    MST = SST / DFT
    MSE = SSE / DFE
    F = MST / MSE
    pvalue = 1 - f.cdf(F,DFT,DFE)

    ANOVA_Table.loc['Treatment','MS'] = MST
    ANOVA_Table.loc['Error','MS'] = MSE

    # Print ANOVA table
    print('\nANOVA Table')
    print(ANOVA_Table)
    print(f'\nF-statistic      : {F}')
    print(f'p-value          : {pvalue}')
    if pvalue > Alpha:
        print('p-value > alpha  : Treatment is not significant')
    else:
        print('p-value <= alpha : Treatment is significant')

    # Build confidence interval table
    CI_Table = pd.DataFrame()
    Index = 0
    for i in range(len(Columns)):
        for j in range(len(Columns)):
            if j>i:

                # Store variable tested
                CI_Table.loc[Index,'Variables'] = f'{Columns[i]} - {Columns[j]}'

                # Difference in means
                y_d = y_i[j] - y_i[i]

                # Compute CI interval
                t_value = np.array(t.interval(1-Alpha,DFE))
                CI = y_d + t_value * np.sqrt(MSE * (1/len(y[i]) + 1/len(y[j])))

                # Store results and increment
                CI_Table.loc[Index,f'{int((1-Alpha)*100)}% CI Low'] = CI[0]
                CI_Table.loc[Index,f'{int((1-Alpha)*100)}% CI Up'] = CI[1]
                Index += 1

    # Boxplot of the results
    Width = 2.5 + len(y)
    Figure, Axis = plt.subplots(1,1, dpi=100, figsize=(Width,4.5))
    for i, Array in enumerate(y):

        # Create random positions
        Array = np.sort(Array)
        Norm = norm.pdf(np.linspace(-3,3,len(Array)), scale=1.5)
        Norm = Norm / max(Norm)
        RandPos = np.random.normal(0,0.03,len(Array)) * Norm + i
        Axis.plot(RandPos - RandPos.mean() + i, Array, linestyle='none',
                  marker='o',fillstyle='none', color=(1,0,0), ms=5)
        
        # Plot
        Axis.boxplot(Array, vert=True, widths=0.35,
                    showmeans=True,meanline=True,
                    showfliers=False, positions=[i],
                    capprops=dict(color=(0,0,0)),
                    boxprops=dict(color=(0,0,0)),
                    whiskerprops=dict(color=(0,0,0),linestyle='--'),
                    medianprops=dict(color=(0,1,0)),
                    meanprops=dict(color=(0,0,1)))

    # Add data for legend
    Axis.plot([],linestyle='none',marker='o',fillstyle='none', color=(1,0,0), label='Data')
    Axis.plot([],color=(0,0,1), label='Mean', linestyle='--')
    Axis.plot([],color=(0,1,0), label='Median')
    
    # Set ticks and axes label
    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])
    Axis.set_xticks(np.arange(len(Columns)))
    Axis.set_xticklabels([C.replace(' ','\n') for C in Columns], rotation=0)

    # Annotate with ANOVA values
    CI_Cols = CI_Table.columns
    if len(YLim) == 2:
        Shift = (YLim[1] - YLim[0]) / 8
    else:
        Shift = (max(i.max() for i in y) - min(i.min() for i in y)) / 5
    Indices = {}
    for c in range(len(Columns)):
        Indices[c] = [[],[]]
    for i, Row in CI_Table.iterrows():

        # Get data values
        V1, V2 = Row['Variables'].split(' - ')
        iC, jC = list(Columns).index(V1), list(Columns).index(V2)
        yi, yj = y[iC], y[jC]

        # Plot line
        YLine = 1.05 * max(yi.max(), yj.max())

        if jC - iC == 1 and YLine not in Indices[iC][0] and YLine not in Indices[jC][1]:
            if len(Indices[jC][1]) > 0:
                if min(Indices[jC][1]) - YLine < Shift:
                    YLine = min(Indices[jC][1]) + Shift
            Indices[iC][0].append(YLine)
            Indices[jC][1].append(YLine)

        else:
            if len(Indices[iC][0]) > 0:
                if YLine < max(Indices[iC][0]):
                    YLine = max(Indices[iC][0])
                if YLine - max(Indices[iC][0]) < Shift:
                    YLine = max(Indices[iC][0]) + Shift
            if len(Indices[jC][1]) > 0:
                if YLine < max(Indices[jC][1]):
                    YLine = max(Indices[jC][1])
                if YLine - max(Indices[jC][1]) < Shift:
                    YLine = max(Indices[jC][1]) + Shift

            while YLine in Indices[iC][0] or YLine in Indices[jC][1]:
                YLine += Shift

            if YLine not in Indices[iC][0]:
                Indices[iC][0].append(YLine)
            if YLine not in Indices[jC][1]:
                Indices[jC][1].append(YLine)


        Plot = Axis.plot([iC+0.05, jC-0.05], [YLine, YLine], color=(0,0,0),
                         marker='|', linewidth=0.5)
        MarkerSize = Plot[0].get_markersize()
        
        # Mark significance level
        if pvalue < 0.001:
            Text = 'p-value < 0.001'
        elif pvalue < 0.01:
            Text = 'p-value < 0.01' 
        elif pvalue < 0.05:
            Text = 'p-value < 0.05'
        else:
            Text = 'p-value > 0.05'
        Axis.annotate(Text, xy=[(len(Columns)-1)/2, 1], ha='center')

        # Write confidence interveal
        CI1 = round(Row[CI_Cols[1]],1)
        CI2 = round(Row[CI_Cols[2]],1)
        Text = f'CI ({CI1},{CI2})'

        Axis.annotate(Text, xy=[(iC + jC)/2, YLine], ha='center',
                        xytext=(0, 1.2*MarkerSize), textcoords='offset points',)
        
        # Store max line y position
        if i == 0:
            Max = YLine*1.05
        else:
            Max = max([Max, YLine + MarkerSize * 0.25])


    # If custom limits
    if len(YLim) == 2:
            Axis.set_ylim(YLim)
    else:
        Min = min([min(i)for i in y]) - MarkerSize * 0.125
        Axis.set_ylim([Min, Max])

    # Add legend
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.125))
    plt.subplots_adjust(left=0.25, right=0.75)

    # Save figure if name given
    if len(FigName) > 0:
        plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=196)
    plt.show(Figure)

    return CI_Table

#%% Main
# Main part

def Main(Alpha=0.05):

    # Record time elapsed
    Time.Process(1, 'Segmentation stats')

    # Set paths
    Dirs = SetDirectories('FEXHIP-Histology')
    DataDir = Dirs['Data']
    ResDir = Dirs['Results'] / '04_Statistics'
    Path.mkdir(ResDir, exist_ok=True)

    # Load data
    DonorsData = pd.read_excel(DataDir / 'DonorsList.xlsx', index_col=[0])
    Data = pd.read_csv(ResDir / 'SegmentationData.csv', header=[0,1], index_col=[0,1,2,3])
    HCHC_Distribution = pd.read_csv(ResDir / 'HCHC_Distances.csv', index_col=[0,1,2,3])
    OSOS_Distribution = pd.read_csv(ResDir / 'OSOS_Distances.csv', index_col=[0,1,2,3])
    HCOS_Distribution = pd.read_csv(ResDir / 'HCOS_Distances.csv', index_col=[0,1,2,3])
    HCCL_Distribution = pd.read_csv(ResDir / 'HCCL_Distances.csv', index_col=[0,1,2,3])
    HC_Distribution = pd.read_csv(ResDir / 'HC_Distances.csv', index_col=[0,1,2,3])
    OS_Distribution = pd.read_csv(ResDir / 'OS_Distances.csv', index_col=[0,1,2,3])
    CL_Distribution = pd.read_csv(ResDir / 'CL_Distances.csv', index_col=[0,1,2,3])
    ImageQuality = pd.read_csv(ResDir / 'ImageQuality.csv', header=[0,1], index_col=[0,1,2,3])
    OS_Colors = pd.read_csv(ResDir / 'OS_Colors.csv', header=[0,1], index_col=[0,1,2,3])

    # Replace sex letter by numeric dummy variables
    Dict = {'M':0,'F':1}
    DonorsData['Sex (-)'] = DonorsData['Sex (-)'].replace(Dict)

    # Add sex and age data
    Data[('Data','Sex (-)')] = np.nan
    Data[('Data','Age (year)')] = np.nan
    for Idx in Data.index:
        Donor = int(Idx[0])
        if Donor in DonorsData.index:
            Data.loc[Idx,('Data','Sex (-)')] = DonorsData.loc[Donor,'Sex (-)']
            Data.loc[Idx,('Data','Age (year)')] = DonorsData.loc[Donor,'Age (year)']

    # Drop nan values (no sample) and get samples statistics
    ImageQuality = ImageQuality.dropna(axis=0, how='any')
    Data = Data.loc[ImageQuality.index]
    DataStats(Data)

    # Investigate images sharpness
    FName = ResDir / 'Filter_Sharpness'
    BoxPlot([ImageQuality['Sharpness']], Labels=['','Sharpness (-)'])
    ShowQuality(ImageQuality, [('Sharpness','Value')], Dirs, FigSave='')

    # Filter out blurry images
    S = ImageQuality[('Sharpness','Value')]
    IQR = S.quantile(0.75) - S.quantile(0.25)
    MinS = S.quantile(0.25) - IQR * 1.5
    S_Filter = ImageQuality[('Sharpness','Value')] > MinS
    BoxPlot([ImageQuality[S_Filter][('Sharpness','Value')]], Labels=['','Sharpness (-)'])
    ShowQuality(ImageQuality[S_Filter], [('Sharpness','Value')], Dirs, FigSave='')
    ImageQuality = ImageQuality[S_Filter].sort_index()

    # Investigate RGB data
    TrainingPath = Dirs['Data'] / 'Training'
    Names = CVAT.GetData(TrainingPath)[0]
    CommonROI = CVAT.CommonROI()[1]
    Ref = np.zeros((3, 255))
    for i in range(3):
        Ref[i] = np.histogram(CommonROI[...,i], bins=255, range=[0,255])[0]
    Ref = Ref / 478**2

    # Get RGB distribution difference from reference image
    Histograms = ImageQuality[['R','G','B']].values
    Histograms = np.reshape(Histograms, (len(ImageQuality), 3, 255))
    Histograms = Histograms / 478**2
    Differences = np.linalg.norm(Histograms - Ref, axis=1)
    ImageQuality['Distances'] = np.sum(Differences, axis=-1)
    BoxPlot([ImageQuality['Distances']], Labels=['','Distances (-)'])
    ShowQuality(ImageQuality, ['Distances'], Dirs, NValues=7, FigSave='')
    
    # Filter out too strongly stained images
    N_Filter = ImageQuality['Distances'] < ImageQuality['Distances'].quantile(0.75)
    BoxPlot([ImageQuality[N_Filter]['Distances']], Labels=['','Distances (-)'])
    ShowQuality(ImageQuality[N_Filter], ['Distances'], Dirs, NValues=7, FigSave='')
    ImageQuality = ImageQuality[N_Filter].sort_index()

    # Filter data
    Indices = ImageQuality.index
    Data = Data.loc[Indices]
    HCHC_Distribution = HCHC_Distribution.loc[Indices]
    HCOS_Distribution = HCOS_Distribution.loc[Indices]
    HCCL_Distribution = HCCL_Distribution.loc[Indices]
    OSOS_Distribution = OSOS_Distribution.loc[Indices]
    HC_Distribution = HC_Distribution.loc[Indices]
    OS_Distribution = OS_Distribution.loc[Indices]
    CL_Distribution = CL_Distribution.loc[Indices]
    OS_Colors = OS_Colors.loc[Indices]

    # Look at osteocytes colors
    Histograms = OS_Colors[['R','G','B']].values
    Histograms = np.reshape(Histograms, (len(OS_Colors), 3, 255))
    Histograms = (Histograms.T / np.sum(Histograms, axis=-1).T).T
    Histograms = np.sum(Histograms, axis=0)

    NC = OS_Colors / OS_Colors.sum(axis=0)['R'].sum()
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(np.arange(255), Histograms[0], color=(1,0,0), label='R channel')
    Axis.plot(np.arange(255), Histograms[1], color=(0,1,0), label='G channel')
    Axis.plot(np.arange(255), Histograms[2], color=(0,0,1), label='B channel')
    Axis.set_xlabel('Pixel Value (-)')
    Axis.set_ylabel('Relative distribution (-)')
    plt.legend()
    plt.show(Figure)

    # Keep only data with diaphysis, neck superior and inferior
    DF = Data.groupby(level=[0,1])
    Drop = []
    for Idx, Df in DF:
        L = len(Df.index.unique(level=2))
        if L < 3:
            Drop.append(Idx[0])
    DataSites = Data.drop(index=Drop, level=0)
    DataStats(DataSites)

    # Keep only data with left and right
    DF = Data.groupby(level=0)
    Drop = []
    for Idx, Df in DF:
        L = len(Df.index.unique(level=1))
        if L < 2:
            Drop.append(Idx)
    DataSide = Data.drop(index=Drop, level=0)
    DataStats(DataSide.dropna(axis=1))

    # Linear regression (OLS) model - H vs O
    X = Data[('Haversian canals','Density (%)')]
    Y = Data[('Osteocytes','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')

    FName = ResDir / 'OLS_HC_OS'
    Labels = ['Haversian canals density (%)', 'Osteocytes density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - H vs C  
    X = Data[('Haversian canals','Density (%)')]
    Y = Data[('Cement lines','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')

    FName = ResDir / 'OLS_HC_CL'
    Labels = ['Haversian canals density (%)', 'Cement lines density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - C vs O
    X = Data[('Cement lines','Density (%)')]
    Y = Data[('Osteocytes','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')

    FName = ResDir / 'OLS_CL_OS'
    Labels = ['Cement lines density (%)','Osteocytes density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - H vs O#
    X = 100 - Data[('Haversian canals','Density (%)')].values.astype('float')
    Y = np.array(Data[('Osteocytes','Number (-)')],float) / (0.5**2 * X/100)
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')

    FName = ResDir / 'OLS_HC_OSn'
    Labels = ['BA/TA (%)',r'O$_{N}$/BA (mm$^{-2}$)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - H vs Age
    X = Data[('Data','Age (year)')]
    Y = Data[('Haversian canals','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')
    DataFrame = DataFrame.dropna()

    FName = ResDir / 'OLS_Age_HC'
    Labels = ['Donor Age (year)', 'Haversian canals density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - O vs Age
    X = Data[('Data','Age (year)')]
    Y = Data[('Osteocytes','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')
    DataFrame = DataFrame.dropna()

    FName = ResDir / 'OLS_Age_OS'
    Labels = ['Donor Age (year)', 'Osteocytes density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Linear regression (OLS) model - H vs Age
    X = Data[('Data','Age (year)')]
    Y = Data[('Cement lines','Density (%)')]
    DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    DataFrame['Donor'] = Data.index.get_level_values('Donor')
    DataFrame['Site'] = Data.index.get_level_values('Site')
    DataFrame = DataFrame.dropna()

    FName = ResDir / 'OLS_Age_CL'
    Labels = ['Donor Age (year)', 'Cement lines density (%)']
    PlotOLSDonors(DataFrame,Labels,FName=FName)

    # Morphometric analysis
    DataStats(Data)
    PrintMorphoStats(Data)

    # Show distances distribution - average curve and envelope
    Labels = ['Distance from a Haversian canal ($\mu m$)', 'Fraction of bone (%)']
    FName = ResDir / 'Distances_HCBone'
    HC_D50, HC_D95 = PlotDistribution(HC_Distribution, Labels, XLim=[0, 350], FName=FName)

    Labels = ['Distance from an osteocyte ($\mu m$)', 'Fraction of bone (%)']
    FName = ResDir / 'Distances_OSBone'
    OS_D50, OS_D95 = PlotDistribution(OS_Distribution, Labels, XLim=[0, 100], FName=FName)

    Labels = ['Distance from a cement line ($\mu m$)', 'Fraction of bone (%)']
    FName = ResDir / 'Distances_CLBone'
    CL_D50, CL_D95 = PlotDistribution(CL_Distribution, Labels, XLim=[0, 100], FName=FName)

    # Inter-segment distances
    Labels = ['Maximum distance from a Haversian canal ($\mu m$)', 'Fraction of bone (%)']
    FName = ResDir / 'Distances_InterHC'
    HCHC_D50, HCHC_D95 = PlotDistribution(HCHC_Distribution.dropna(), Labels, FName=FName)

    Labels = ['Maximum distance from an osteocyte ($\mu m$)', 'Fraction of bone (%)']
    FName = ResDir / 'Distances_InterOS'
    OSOS_D50, OSOS_D95 = PlotDistribution(OSOS_Distribution, Labels, XLim=[0, 200], FName=FName)

    # Distribution of osteocytes from Haversian canals
    Labels = ['Distance from Haversian canal ($\mu m$)', 'Fraction of osteocytes (%)']
    FName = ResDir / 'Distances_HCOS'
    HCOS_D50, HCOS_D95 = PlotDistribution(HCOS_Distribution, Labels, FName=FName)

    # Distribution of cement lines from Haversian canals
    Labels = ['Distance from Haversian canal ($\mu m$)', 'Fraction of cement lines (%)']
    FName = ResDir / 'Distances_HCCL'
    HCCL_D50, HCCL_D95 = PlotDistribution(HCCL_Distribution, Labels, FName=FName)

    # BoxPlot([D50HC, D50OS, D50CL], Labels=['50% of pixels', 'Distance from Haversian canal ($\mu m$)'],
    #         SetsLabels=['Haversian\ncanal', 'Osteocytes','Cement\nlines'])
    # BoxPlot([D95HC, D95OS, D95CL], Labels=['95% of pixels', 'Distance from Haversian canal ($\mu m$)'],
    #         SetsLabels=['Haversian\ncanal', 'Osteocytes','Cement\nlines'])



    # Get mean values
    Means = Data.groupby(level=[0,1,2]).mean()
    Std = Data.groupby(level=[0,1,2]).std(ddof=1)

    # Draw general boxplot
    A1, A2, A3 = [], [], []
    Df = Data.groupby(level=[0,1,2])
    for df in Df:
        Area = df[1].loc[df[0]].index.max() * 0.5**2
        if df[0][2] == 'Diaphysis':
            A1.append(Area)
        elif df[0][2] == 'Neck Inferior':
            A2.append(Area)
        else:
            A3.append(Area)

    BoxPlot([A1, A2, A3], Labels=['Area Analysed (mm$^2$)',''],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'])

    # Draw boxplots for laterality
    Values = Means[('Haversian canals','Density (%)')]
    ArrayList = [Values.loc[:,'Left'],
                 Values.loc[:,'Right']]
    FName = ResDir / 'Side_HC'
    BoxPlot(ArrayList, Labels=['', r'H$_{\rho}$ (%)'],
            SetsLabels=['Left','Right'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    Values = Means[('Osteocytes','Density (%)')]
    ArrayList = [Values.loc[:,'Left'],
                 Values.loc[:,'Right']]
    FName = ResDir / 'Side_OS'
    BoxPlot(ArrayList, Labels=['', r'O$_{\rho}$ (%)'],
            SetsLabels=['Left','Right'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    Values = Means[('Cement lines','Density (%)')]
    ArrayList = [Values.loc[:,'Left'],
                 Values.loc[:,'Right']]
    FName = ResDir / 'Side_CL'
    BoxPlot(ArrayList, Labels=['', r'C$_{\rho}$ (%)'],
            SetsLabels=['Left','Right'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    # Perform ANOVAs for laterality
    Values = Means[('Haversian canals','Density (%)')]
    FName = str(ResDir / 'Side_HC')
    ANODATA = pd.DataFrame()
    ANODATA['Left'] = Values.loc[:,'Left']
    ANODATA['Right'] = Values.loc[:,'Right']
    ANOVA(ANODATA, Labels=['', r'H$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)

    Values = Means[('Osteocytes','Density (%)')]
    FName = str(ResDir / 'Side_OS')
    ANODATA = pd.DataFrame()
    ANODATA['Left'] = Values.loc[:,'Left']
    ANODATA['Right'] = Values.loc[:,'Right']
    ANOVA(ANODATA, Labels=['', r'O$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)

    Values = Means[('Cement lines','Density (%)')]
    FName = str(ResDir / 'Side_CL')
    ANODATA = pd.DataFrame()
    ANODATA['Left'] = Values.loc[:,'Left']
    ANODATA['Right'] = Values.loc[:,'Right']
    ANOVA(ANODATA, Labels=['', r'C$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)

    # Draw boxplots for sex
    xValues = Means[('Data','Sex (-)')].values
    yValues = Means[('Haversian canals','Density (%)')]
    ArrayList = [yValues[xValues == 0],
                 yValues[xValues == 1]]
    FName = ResDir / 'Sex_HC'
    BoxPlot(ArrayList, Labels=['', r'H$_{\rho}$ (%)'],
            SetsLabels=['M','F'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    xValues = Means[('Data','Sex (-)')].values
    yValues = Means[('Osteocytes','Density (%)')]
    ArrayList = [yValues[xValues == 0],
                 yValues[xValues == 1]]
    FName = ResDir / 'Sex_OS'
    BoxPlot(ArrayList, Labels=['', r'O$_{\rho}$ (%)'],
            SetsLabels=['M','F'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    xValues = Means[('Data','Sex (-)')].values
    yValues = Means[('Cement lines','Density (%)')]
    ArrayList = [yValues[xValues == 0],
                 yValues[xValues == 1]]
    FName = ResDir / 'Sex_CL'
    BoxPlot(ArrayList, Labels=['', r'C$_{\rho}$ (%)'],
            SetsLabels=['M','F'], Ttest='Ind', YLim=[0,20],
            FigName=FName)
    
    # Perform ANOVAs for sex
    Sex = Means[('Data','Sex (-)')].values
    Values = Means[('Haversian canals','Density (%)')]
    FName = str(ResDir / 'Sex_HC')
    ANODATA = pd.DataFrame()
    ANODATA['M'] = Values[Sex == 0]
    ANODATA['F'] = Values[Sex == 1]
    ANOVA(ANODATA, Labels=['', r'H$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)

    Values = Means[('Osteocytes','Density (%)')]
    FName = str(ResDir / 'Sex_OS')
    ANODATA = pd.DataFrame()
    ANODATA['M'] = Values[Sex == 0]
    ANODATA['F'] = Values[Sex == 1]
    ANOVA(ANODATA, Labels=['', r'O$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)

    Values = Means[('Cement lines','Density (%)')]
    FName = str(ResDir / 'Sex_CL')
    ANODATA = pd.DataFrame()
    ANODATA['M'] = Values[Sex == 0]
    ANODATA['F'] = Values[Sex == 1]
    ANOVA(ANODATA, Labels=['', r'C$_{\rho}$ (%)'], YLim=[0,22], FigName=FName)
    
    # Draw boxplots for different anatomical sites
    Values = Means[('Haversian canals','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_HC'
    BoxPlot(ArrayList, Labels=['', r'H$_{\rho}$ (%)'], YLim=[0,20],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)
    
    Values = Means[('Osteocytes','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_OS'
    BoxPlot(ArrayList, Labels=['', r'O$_{\rho}$ (%)'], YLim=[0,20],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)

    Values = Means[('Cement lines','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_CL'
    BoxPlot(ArrayList, Labels=['', r'C$_{\rho}$ (%)'], YLim=[0,20],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)

    # Perform ANOVAs for anatomical sites
    Values = Means[('Haversian canals','Density (%)')]
    FName = str(ResDir / 'Site_HC')
    ANODATA = pd.DataFrame()
    ANODATA['Diaphysis'] = Values.loc[:,:,'Diaphysis']
    ANODATA['Neck Inferior'] = Values.loc[:,:,'Neck Inferior']
    ANODATA['Neck Superior'] = Values.loc[:,:,'Neck Superior']
    ANOVA(ANODATA, Labels=['', r'H$_{\rho}$ (%)'], YLim=[0,23.5], FigName=FName)

    Values = Means[('Osteocytes','Density (%)')]
    FName = str(ResDir / 'Site_OS')
    ANODATA = pd.DataFrame()
    ANODATA['Diaphysis'] = Values.loc[:,:,'Diaphysis']
    ANODATA['Neck Inferior'] = Values.loc[:,:,'Neck Inferior']
    ANODATA['Neck Superior'] = Values.loc[:,:,'Neck Superior']
    ANOVA(ANODATA, Labels=['', r'O$_{\rho}$ (%)'], YLim=[0,23.5], FigName=FName)

    Values = Means[('Cement lines','Density (%)')]
    FName = str(ResDir / 'Site_CL')
    ANODATA = pd.DataFrame()
    ANODATA['Diaphysis'] = Values.loc[:,:,'Diaphysis']
    ANODATA['Neck Inferior'] = Values.loc[:,:,'Neck Inferior']
    ANODATA['Neck Superior'] = Values.loc[:,:,'Neck Superior']
    ANOVA(ANODATA, Labels=['', r'C$_{\rho}$ (%)'], YLim=[0,23.5], FigName=FName)

    # Draw boxplots for different anatomical sites variability
    Values = Std[('Haversian canals','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_Std_HC'
    BoxPlot(ArrayList, Labels=['', 'Haversian Canals Density (%)'], YLim=[0,7.2],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)
    
    Values = Std[('Osteocytes','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_Std_OS'
    BoxPlot(ArrayList, Labels=['', 'Osteocytes Density (%)'], YLim=[0,7.2],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)

    Values = Std[('Cement lines','Density (%)')]
    ArrayList = [Values.loc[:,:,'Diaphysis'],
                 Values.loc[:,:,'Neck Inferior'],
                 Values.loc[:,:,'Neck Superior']]
    FName = ResDir / 'Site_Std_CL'
    BoxPlot(ArrayList, Labels=['', 'Cement Lines Density (%)'], YLim=[0,7],
            SetsLabels=['Diaphysis','Neck\nInferior','Neck\nSuperior'], Ttest='Ind',
            FigName=FName)

    # Correlate densities
    Xc = np.array(Data[('Haversian canals','Density (%)')],float)
    Yc = np.array(Data[('Cement lines','Density (%)')],float)
    FName = ResDir / 'OLS_HC_CL'
    PlotOLS(Xc, Yc, Labels=['Haversian Canals Density (%)', 'Cement Lines Density (%)'], FName=FName)

    Xc = np.array(Data[('Haversian canals','Density (%)')],float)
    Yc = np.array(Data[('Osteocytes','Density (%)')],float)
    FName = ResDir / 'OLS_HC_OS'
    PlotOLS(Xc, Yc, Labels=['Haversian Canals Density (%)', 'Osteocytes Density (%)'], FName=FName)

    Xc = np.array(Data[('Cement lines','Density (%)')],float)
    Yc = np.array(Data[('Osteocytes','Density (%)')],float)
    FName = ResDir / 'OLS_CL_OS'
    PlotOLS(Xc, Yc, Labels=['Cement Lines Density (%)', 'Osteocytes Density (%)'], FName=FName)

    # Additional correlation
    Xc = 1 - np.array(Data[('Haversian canals','Density (%)')],float) / 100
    Yc = np.array(Data[('Osteocytes','Number (-)')],float) / (0.5**2 * Xc)
    FName = ResDir / 'OLS_BVTV_OS'
    PlotOLS(Xc, Yc, Labels=['BV/TV (-)', 'Osteocytes Density (mm$^{-2}$)'], FName=FName)

    # # Mixed-LM
    # X = 100 - LMEData[('Haversian canals','Density (%)')].values.astype('float')
    # Y = np.array(LMEData[('Osteocytes','Number (-)')],float) / (0.5**2 * X/100)
    # DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    # DataFrame['Donor'] = [int(D) for D in LMEData.index.get_level_values('Donor')]
    # DataFrame['Site'] = LMEData.index.get_level_values('Site')
    # DataFrame['Age'] = LMEData[('Data', 'Age (year)')].values

    # LME = smf.mixedlm('Y ~ X - 1',
    #                 data=DataFrame,
    #                 groups=DataFrame['Donor'],
    #                 re_formula='0 + Donor',
    #                 )
    # Fit = LME.fit(reml=True, method='lbfgs')
    # print(Fit.summary())

    # FName = ResDir / 'LME_HC_OS'
    # PlotMixedLM(DataFrame, Fit, LME, Xlabel='BA/TA (%)',
    #             Ylabel='Osteocytes Density (mm$^{-2}$)', FigName=FName)

    # DataFrame.index = Data.index
    # ShowQuality(DataFrame, ['Y'], Dirs, NValues=7, FigSave='')


    # BoxPlot([rex], Labels=['', 'Donor Random Effect (mm$^{-2}$)'])
    # DataFrame['e'] = Fit.resid
    # BoxPlot([DF[1]['e'] for DF in DataFrame.groupby('Donor')],
    #         SetsLabels=[DF[0] for DF in DataFrame.groupby('Donor')],
    #         Labels=['Donor (-)', 'Residuals (mm$^{-2}$)'],
    #         FigName = ResDir / 'LME_QQ_Re')


    # # The BLUPs
    # re = Fit.random_effects
    # # Multiply each BLUP by the random effects design matrix for one group
    # rex = [np.dot(LME.exog_re_li[j][0], re[k]) for (j, k) in enumerate(LME.group_labels)]
    # QQPlot(rex,FigName = ResDir / 'LME_QQ_Re')

    # # Test for donor random effect
    # Fit2 = smf.ols('Y ~ X',
    #                 data=DataFrame,
    #                 ).fit(reml=True)

    # # Log-Likelihood Ratio Test
    # LRT = -2 * (Fit2.llf - Fit.llf)
    # delta_df = 1
    # p = 1 - chi2.cdf(LRT,delta_df)
    # print('p value for LME is preferable over OLS')
    # print(p)
    # print('If p < 0.05: random effect is significant')
    
    # # Correlate with age
    # X = Data[('Data','Age (year)')].values.astype('float')
    # Y = Data[('Haversian canals','Density (%)')].values.astype('float')
    # DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    # DataFrame['Donor'] = [int(D) for D in Data.index.get_level_values('Donor')]
    # DataFrame['Sample'] = Data.index.get_level_values('Side')
    # LME = smf.mixedlm('Y ~ X',
    #                 data=DataFrame, groups=DataFrame['Donor'],
    #                 re_formula='1', vc_formula={'Sample': '0 + Sample'}
    #                 ).fit(reml=False, method=['lbfgs'])
    # FName = ResDir / 'LME_Age_HC'
    # PlotMixedLM(DataFrame, LME, Xlabel='Age (year)',
    #             Ylabel='Haversian canals Density (%)', FigName=FName)

    # X = Data[('Data','Age (year)')].values.astype('float')
    # Y = Data[('Osteocytes','Density (%)')].values.astype('float')
    # DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    # DataFrame['Donor'] = [int(D) for D in Data.index.get_level_values('Donor')]
    # DataFrame['Sample'] = Data.index.get_level_values('Side')
    # LME = smf.mixedlm('Y ~ X',
    #                 data=DataFrame, groups=DataFrame['Donor'],
    #                 re_formula='1', vc_formula={'Sample': '0 + Sample'}
    #                 ).fit(reml=False, method=['lbfgs'])
    # FName = ResDir / 'LME_Age_OS'
    # PlotMixedLM(DataFrame, LME, Xlabel='Age (year)',
    #             Ylabel='Osteocytes Density (%)', FigName=FName)

    # X = Data[('Data','Age (year)')].values.astype('float')
    # Y = Data[('Cement lines','Density (%)')].values.astype('float')
    # DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
    # DataFrame['Donor'] = [int(D) for D in Data.index.get_level_values('Donor')]
    # DataFrame['Sample'] = Data.index.get_level_values('Side')
    # LME = smf.mixedlm('Y ~ X',
    #                 data=DataFrame, groups=DataFrame['Donor'],
    #                 re_formula='1', vc_formula={'Sample': '0 + Sample'}
    #                 ).fit(reml=False, method=['lbfgs'])
    # FName = ResDir / 'LME_Age_CL'
    # PlotMixedLM(DataFrame, LME, Xlabel='Age (year)',
    #             Ylabel='Cement lines Density (%)', FigName=FName)




    # # Build p-values matrix
    # Corr = np.zeros((len(Data.columns), len(Data.columns)))
    # for iX, Xc in enumerate(Data.columns):
    #     for iY, Yc in enumerate(Data.columns):
    #         if iX <= iY:
    #             Corr[iX,iY] = np.nan
    #         else:
    #             LME = Results[(Xc, Yc)]
    #             Corr[iX,iY] = round(LME.pvalues[1],3)

    # # Categorise p-values
    # Cat = Corr.copy()
    # Cat[Cat >= Alpha] = 4
    # Cat[Cat < 0.001] = 1
    # Cat[Cat < 0.01] = 2
    # Cat[Cat < 0.05] = 3

    # # Plot p-values categories
    # Labels = [C[0] + '\n' + C[1] for C in Data.columns]
    # Figure, Axis = plt.subplots(1,1, figsize=(9,12))
    # Im = Axis.matshow(Cat, vmin=0, vmax=4.5, cmap='binary')
    # Axis.xaxis.set_ticks_position('bottom')
    # Axis.set_xticks(np.arange(len(Corr))-0.5, minor=True)
    # Axis.set_xticks(np.arange(len(Corr)))
    # Axis.set_xticklabels(Labels, ha='center', rotation=90)
    # Axis.set_yticks(np.arange(len(Corr))-0.5, minor=True)
    # Axis.set_yticks(np.arange(len(Corr)))
    # Axis.set_yticklabels(Labels)
    # Axis.grid(which='minor', color=(1,1,1), linestyle='-', linewidth=2)
    # Cb = plt.colorbar(Im, ticks=[1, 2, 3, 4], fraction=0.046, pad=0.04, values=np.linspace(1,4,100))
    # Cb.ax.set_yticklabels(['<0.001', '<0.01', '<0.05','$\geq$0.05'])
    # Figure.savefig(StatDir / 'Pvalues.png', dpi=Figure.dpi, bbox_inches='tight')
    # plt.show(Figure)


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
