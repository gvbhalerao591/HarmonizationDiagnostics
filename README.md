# DiagnoseHarmonize version 0.0.1

DiagnoseHarmonize is a library developed for the streamline application and assesment of harmonization algorithms at the summary measure level, as well as the establishment of a centralised location for popular existing harmonization methods that are well validated within the literature.

If you find any issues or bugs in any part of the code, please raise it as an issue or alternatively contact the following:

Jake Turnbull: [jacob.turnbull@ndcn.ox.ac.uk](mailto:jacob.turnbull@ndcn.ox.ac.uk)

Gaurav Bhalerao: [Gaurav.bhalerao@ndcn.ox.ac.uk](Gaurav.bhalerao@ndcn.ox.ac.uk)

# Overview

This package contains a set of different statistical tests defined in DiagnosticFunctions.py as well as complementry plotting functions for each case defined in PlotDiagnosticResults.py as well as a set of well established harmonization methods from the literature, which can be found in HarmonizationFunctions.py.

This package contains several different modules, each defined for different parts of harmonization pipelines. The first is DiagnosticFunctions.py, which contains a set of statistical tests for the assesment of batch and covariate effects within your data. The second is a complementary module which provides specialised plotting functions for the visualisation of each of these these tests. We also include a set of harmonization function stored in HarmonizationFunctions.py which contain well validated harmonization methods for summary level data.

In each case, after proper installation, a specific module from the library can be loaded as from DiagnoseHarmonize import DiagnosticFunctions.py. The individual functions can then be called by from the loaded module, e.g DiagnosticFunctions.PcaCorr

The main application of this library was originally intended to be the use of pre-made tools for the direct reporting of batch/site effects in both cross-sectional and longitudinal datasets, but has since been extended to serve as a central location for new harmonization methods and the assembly or reporting pipelines for MRI harmonization.

Current pre-built harmonization pipelines are stored in the DiagnosticReport.py module.

# DiagnosticFunctions

# PlotDiagnosticResults

# HarmonizationFunctions

# DiagnosticReport

# LoggingTool