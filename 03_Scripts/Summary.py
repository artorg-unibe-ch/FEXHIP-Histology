#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script collects results from the statistical
    analysis, write and build a corresponding report
    using pylatex

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2023
    """

#%% Imports
# Modules import

import argparse
import pandas as pd
from Utils import Time
from pathlib import Path

from pylatex import Document, Section, Subsection
from pylatex import Figure, SubFigure, NoEscape, NewPage, Command
from pylatex.package import Package

#%% Functions
# Define functions

def TableText(ParamsList):

    # Define text parts
    Start = r'''
\begin{center}
\begin{tabular}{lrrrrrr}
\hline
           &  Coef. & [0.025 & 0.975] & P$> |$z$|$ \\
\hline
'''

    Core = '''
{V}        &  {X} &  {CIl} &  {CIu} &  {p}  '''

    End = r'''
\hline
\end{tabular}
\end{center}
\end{table}
'''
        
    # Build text
    Text = Start
    for (M, V), R in ParamsList.iterrows():
        X, CIl, CIu, p = R
        if not pd.isna(M):
            V = M + ' ' + V
        if '%' in V:
            V = V.replace('%','\%')
        Context = {'V': V,
                   'X': round(X,3),
                   'p': round(p,3),
                   'CIl': round(CIl,3),
                   'CIu': round(CIu,3)}
        Text += Core.format(**Context) + r'\\'
    Text += End

    return Text

def TableMorpho(ParamsList):

    # Define text parts
    Start = r'''
\begin{center}
\begin{tabular}{lrrrrrr}
\hline
           &  Coef. & [0.025 & 0.975] & P$> |$z$|$ \\
\hline
'''

    Core = '''
{V}        &  {X} &  {CIl} &  {CIu} &  {p}  '''

    End = r'''
\hline
\end{tabular}
\end{center}
\end{table}
'''
        
    # Build text
    Text = Start
    for V, R in ParamsList.iterrows():
        X, CIl, CIu, p = R
        Context = {'V': V,
                   'X': round(X,3),
                   'p': round(p,3),
                   'CIl': round(CIl,3),
                   'CIu': round(CIu,3)}
        Text += Core.format(**Context) + r'\\'
    Text += End

    return Text


#%% Main
# Main part

def Main():

    # Record time elapsed
    Time.Process(1, 'Summary report')

    # Set paths
    MainDir = Path(__file__).parent / '..'
    ResDir = MainDir / '03_Results'
    StatDir = ResDir / 'Statistics'

    # Read statistical analysis results
    OsStats = pd.read_csv(ResDir / ('Osteocytes_Statistics.csv'), index_col=[0,1])
    HcStats = pd.read_csv(ResDir / ('Haversian_canals_Statistics.csv'), index_col=[0,1])
    ClStats = pd.read_csv(ResDir / ('Cement_lines_Statistics.csv'), index_col=[0,1])
    Morpho = pd.read_csv(ResDir / ('MorphoCorr.csv'), index_col=[2,3])
    gStats = [ClStats, HcStats, OsStats]
    for S in [OsStats, HcStats, ClStats]:
        S.index.names = ['Method','Variable']

    # Generate report
    Doc = Document(default_filepath=str(ResDir / 'Results'))
    Tissues = ['Cement lines', 'Haversian canals', 'Osteocytes']
    for Tissue, Stats in zip(Tissues,gStats):

        # Create section for tissue type
        Sec = 'Final ' + Tissue
        with Doc.create(Section(Sec, numbering=False)):

            # # Add tissue normality check
            # with Doc.create(Figure(position='h!')) as Fig:
            #     Doc.append(Command('centering'))
                
            #     SubFig = SubFigure(position='b', width=NoEscape(r'0.55\linewidth'))
            #     Image = str(StatDir / Tissue / 'Density_Hist.png')
            #     with Doc.create(SubFig) as SF:
            #         Doc.append(Command('centering'))
            #         SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #         SF.add_caption('Histogram')

            #     SubFig = SubFigure(position='b', width=NoEscape(r'0.45\linewidth'))
            #     Image = str(StatDir / Tissue / 'Density_QQ.png')
            #     with Doc.create(SubFig) as SF:
            #         Doc.append(Command('centering'))
            #         SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #         SF.add_caption('QQ plot')
            # Fig.add_caption('Normality check')

            # Add summary
            Text = TableText(Stats)
            Doc.append(NoEscape(Text))
            # Doc.append(NewPage())
        
            # for (M,V), R in Stats.iterrows():

            #     # Create subsections for each correlated variable
            #     if pd.isna(M):
            #         SubS = V.split(' (')[0]
            #     else:
            #         SubS = M + ' - ' + V.split(' (')[0]
            #     with Doc.create(Subsection(SubS, numbering=False)):  

            #         # Add LME results
            #         if V.split(' (')[0] == Tissue:
            #             Image = M + '_OLS.png'
            #         else:
            #             Image = V.split(' (')[0] + '_LME.png'

            #         Image = str(StatDir / Tissue / Image)
            #         with Doc.create(Figure(position='h!')) as Fig:
            #             Fig.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #             Fig.add_caption(NoEscape('LME results'))

            #         # Check assumptions: X normal distribution
            #         with Doc.create(Figure(position='h!')) as Fig:
            #             Doc.append(Command('centering'))
                        
            #             SubFig = SubFigure(position='b', width=NoEscape(r'0.55\linewidth'))
                        
            #             Image = Image[:-7] + 'Hist.png'
            #             Image = str(StatDir / Tissue / Image)
            #             with Doc.create(SubFig) as SF:
            #                 Doc.append(Command('centering'))
            #                 SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                 SF.add_caption('Histogram')

            #             SubFig = SubFigure(position='b', width=NoEscape(r'0.45\linewidth'))
            #             Image = Image[:-8] + 'QQ.png'
            #             Image = str(StatDir / Tissue / Image)
            #             with Doc.create(SubFig) as SF:
            #                 Doc.append(Command('centering'))
            #                 SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                 SF.add_caption('QQ plot')

            #         Fig.add_caption('Independent variable distribution')
            #         Doc.append(NewPage())

            #         # Check assumptions: Random effect normal distribution and 0 mean
            #         if V.split(' (')[0] != Tissue:
            #             with Doc.create(Figure(position='h!')) as Fig:
            #                 Doc.append(Command('centering'))
                            
            #                 SubFig = SubFigure(position='t', width=NoEscape(r'0.34\linewidth'))
            #                 Image = Image[:-6] + 'RE.png'
            #                 Image = str(StatDir / Tissue / Image)
            #                 with Doc.create(SubFig) as SF:
            #                     Doc.append(Command('centering'))
            #                     SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                     SF.add_caption('Boxplot')

            #                 SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
            #                 Image = Image[:-6] + 'Group_QQ.png'
            #                 Image = str(StatDir / Tissue / Image)
            #                 with Doc.create(SubFig) as SF:
            #                     Doc.append(Command('centering'))
            #                     SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                     SF.add_caption('Group effect QQ plot')
                        
            #             with Doc.create(Figure(position='h!')) as Fig:
            #                 Doc.append(Command('centering'))
                            
            #                 SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
            #                 Image = Image[:-12] + 'Left_QQ.png'
            #                 Image = str(StatDir / Tissue / Image)
            #                 with Doc.create(SubFig) as SF:
            #                     Doc.append(Command('centering'))
            #                     SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                     SF.add_caption('Sample effect: Left QQ plot')

            #                 SubFig = SubFigure(position='t', width=NoEscape(r'0.5\linewidth'))
            #                 Image = Image[:-7] + 'Right_QQ.png'
            #                 Image = str(StatDir / Tissue / Image)
            #                 with Doc.create(SubFig) as SF:
            #                     Doc.append(Command('centering'))
            #                     SF.add_image(Image, width=NoEscape(r'0.9\linewidth'))
            #                     SF.add_caption('Sample effect: Right QQ plot')

            #             Fig.add_caption('Random effects distribution and 0 mean assumptions')

            #         # Check assumptions: Residuals distribution and 0 mean
            #         with Doc.create(Figure(position='h!')) as Fig:
            #             Doc.append(Command('centering'))

            #             SubFig = SubFigure(position='b', width=NoEscape(r'0.27\linewidth'))
            #             Image = Image[:-12] + 'Residuals.png'
            #             Image = str(StatDir / Tissue / Image)
            #             with Doc.create(SubFig) as SF:
            #                 Doc.append(Command('centering'))
            #                 SF.add_image(Image, width=NoEscape(r'0.8\linewidth'))
            #                 SF.add_caption('Boxplot')

            #             SubFig = SubFigure(position='b', width=NoEscape(r'0.5\linewidth'))
                        
            #             Image = Image[:-13] + 'Residuals_QQ.png'
            #             Image = str(StatDir / Tissue / Image)
            #             with Doc.create(SubFig) as SF:
            #                 Doc.append(Command('centering'))
            #                 SF.add_image(Image, width=NoEscape(r'0.8\linewidth'))
            #                 SF.add_caption('QQ plot')

            #         Fig.add_caption('Residuals distribution and 0 mean assumptions')

            #     Doc.append(NewPage())

    # Create section for morphometric analysis
    Doc.append(NewPage())
    Sec = 'Morphometry'
    with Doc.create(Section(Sec, numbering=False)):
        for Y in Tissues:
            if Y in ['Cement lines']:
                Parameters = ['Thickness (px)']
            else:
                Parameters = ['Area (-)', 'Number (-)']
            for P in Parameters:
                Sec = Y + ' - ' + P
                with Doc.create(Subsection(Sec, numbering=False)):
                    F = Morpho.loc[Y]['Parameter'] == P
                    Data = Morpho.loc[Y][F].drop('Unnamed: 0', axis=1)
                    Data = Data.drop('Parameter', axis=1)
                    # Add summary
                    Text = TableMorpho(Data)
                    Doc.append(NoEscape(Text))   

    Doc.packages.append(Package('caption', options='labelformat=empty'))
    Doc.packages.append(Package('subcaption', options='aboveskip=0pt, labelformat=empty'))
    Doc.generate_pdf(clean_tex=False)  

    Time.Process(0)

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
