# pellegrini_temp

This repo is intended to track the ongoing development of the project. It is in-progress and is subject to change.

Presently, the model works by utilizing 2 models which eventually intersect. The first combines a survival neural network (2 hidden layers w/ timepoint as a parameter) with elastic net regression, intended to mimic the construction of an epigenetic clock, and the second combines pseudoinversion with multiple multivariate regression. These processes are used, in conjunction with other filtering methods, to identify methylation probes and by extension genes significant in aging. These outputs are then overlapped, identifying a small set of highly significant probes.

![](figures/data_analysis_overview.png)

The raw data files that the analysis uses are not available here. The cohorts utilized by the models are depicted below

![](figures/experimental_design.png)

All of the figures used in the manuscript are derived from the code and may be accessed there
