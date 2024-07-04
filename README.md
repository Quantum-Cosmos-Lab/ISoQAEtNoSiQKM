# In Search of Quantum Advantage: Estimating the Number of Shots in Quantum Kernel Methods

The repository contains all the necessary files to replicate the numerical simulations performed in the eponymous work.

## Directories

- ./results/

Contains results of numerical simulations

- ./prepared_data/

Contains prepared twonorm and indian pines datasets

- ./lib/

Contains scripts used for numerical analysis

- ./datasets/

Contains unprepared datasets

## .csv
- classical_resources_n_FLOPS_CPU_cycles.csv

Contains the data about the numerical simulation of kernel entries on a classical computer
Three columns include: n; number of floating point operations; CPU cycles

- sonar_Azure_6.csv

Contains results from the MS Azure quantum resource estimation tool for sonar dataset and $6$ ZZ-feature map repetitions in projected quantum kernel family

## .py files

- kernel_concentration_*

Files used for fidelity quantum kernel estimation

- proj_kernel_concentration_*

Files used for projected quantum kernel estimation

## .ipynb files

- Azure_script.ipynb

Python notebook used for quantum resource estimation with MS Azure tool

- Expressibility_estimates.ipynb

Python notebook with the with the analysis for Figure 7

- kernel_concentration_plots.ipynb
- proj_kernel_concentration_plots.ipynb

Python notebooks with the with the analysis for Figure 4

- QRE_analysis_sonar_ZZ6.ipynb

Python notebook preparing files for Figure 6

- resource_comparison.ipynb

Python notebook with the with the analysis for Figure 6

- Sonar_N_estimation.ipynb

Python notebook with the with the analysis for Figure 5