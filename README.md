# Tabular Data Science BIU 2022 Final Project

Please see below all the relevant information for the submission.<br>
All you need, is to run the relevant notebook, local or in Colab.<br>
The Colab notebook is standalone, it will clone the datasets from this repository.<br>

## Colab Inference Notebook:
https://colab.research.google.com/drive/1Fd-fKZ2BuBxSbxb1K_3E3DUM0VF8x11A?usp=sharing

## Repository structure:
    .
    ├── Tabular_Data_Science_Final_Project.ipynb        # Jupyter inference notebook
    ├── Tabular_Data_Science_Final_Project_report.pdf   # Report
    ├── datasets                                        # Datasets folder
    │   ├── cancer.csv                                  # Dataset 1
    │   ├── houses.csv                                  # Dataset 2
    │   ├── salaries.csv                                # Dataset 3
    │   └── sleeping.csv                                # Dataset 4
    ├── lib_tds_rg                                      # Library package
    │   ├── __init__.py
    │   ├── dataset_handling_utils.py                   # Dataset utilities
    │   ├── empirical_experiments_utils.py              # Empirical experiments utilities
    │   ├── plotting_utils.py                           # Plotting utilities
    │   └── tds_rg_module.py                            # Raviv-Gavriely distribution similarities method module
    └── requirements.txt                                # Pip requirements for the project

## Requirements note:
We were asked to supply only the requirments not shown in the course.<br>
The following are all the libs needed: scipy>=1.7.3, pandas, numpy, seaborn, matplotlib, sklearn, statsmodels.
