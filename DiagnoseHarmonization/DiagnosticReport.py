# Diagnostic report generation using DiagnosticFunctions 
def DiagnosticReport(data, batch,
                    covariates=None,
                        covariate_names=None,
                            save_data=False,
                            save_data_name=None,
                            save_dir=None,
                            report_name=None,
                                SaveArtifacts=False,
                                rep=None,
                                    show=False,
                                    timestamped_reports=True):
    """
    Create a diagnostic report for dataset differences across batches, taking into account covariates
    when relevant.
    The different tests used are all defined in DiagnosticFunctions.py and the plots in PlotDiagnosticResults.py.
    The following tests are included:

    Args:
        Data:
        Batch:
        Covariate: array of values, each column is one covariate
        Covariate_names (List): Names of the covariates in the same order as covariate matrix columns
        save_data (Logical): Save the outputs of each test as a pandas dataframe for test results
        Save_dir (String): File path to the directory in which to save report and images
        SaveArtifacts (Logical): Save plots as PNG images in Save_dir or current directory
        rep: StatsReporter object (defined in LoggingTool.py) to log outputs to Save_dir
        show (Logical): Show plots as they are generated, default is False


    Additive components:
        - Cohen' D test for mean differences (standardized mean difference)
        - Mahalanobis distance test for multivariate mean differences
    
    Multiplicative components:
        - Levene's test for variance differences (set as Brown-Forsythe test for rubustness)
        - Variance ratio test between each unique batch pair 
    
    Both:
        - PCA visualization of data colored by batch
        - PCA correlation with batch and covariates 
        - Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair 

    Args:
    Arguments can be given as a pandas dataframe or as individual numpy arrays.

    As a pandas dataframe have the following columns:
        - 'data': subjects x features (np.ndarray)
        - 'batch': batch labels (np.ndarray)
        - 'covariates': subjects x covariates (np.ndarray), optional
        - 'covariate_names': covariate names (list), optional
    Or as individual numpy arrays:
        - data: subjects x features (np.ndarray)
        - batch: batch labels (np.ndarray)
        - covariates: subjects x covariates (np.ndarray), optional
        - covariate_names: covariate names (list), optional

    Returns:
        - report: a HTML file containing the outputs from each diagnostic function (from DiagnosticFunctions.py) and 
        and the corresponding plots (from PlottingFunctions.py)
        - If SaveArtefacts is set to true, all plots are also saved in the same directory as the report
        - If show is set to true, all plots are also displayed as seperare matplotlib figure windows 
    Raises:
        - ValueError: if Data is not a 2D array or batch is not a
        1D array, or if the number of samples in Data and batch do not match. 
    """
# Import the necessary libraries and functions from diagnostic functions and plotting functions
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults
    from DiagnoseHarmonization.LoggingTool import StatsReporter
    from DiagnoseHarmonization.LoggingTool import set_report_path

# Start the log defined using StatsReporter from LoggingTool.py 
    with StatsReporter(save_artifacts=SaveArtifacts, save_dir=None) as report:
        logger = report.logger

        # Run the diagnostic functions and log their outputs
        logger.info("Starting report generation")

        if save_dir is None:
            logger.info("No save directory specified, saving to current working directory")
            save_dir = os.getcwd()
        else:
            logger.info(f"Saving to directory: {save_dir}")

        if report_name is None:
            report_name = "DiagnosticReport.html"
        else:
            if not report_name.endswith(".html"):
                report_name += ".html"
                
        report_path = set_report_path(report, save_dir, report_name=report_name,timestamp=timestamped_reports)
        # Define variable to work as linebreak, to be improved at a later date


        line_break_in_text = "-----------------------------------------------------------------------------------------------------------------------------"

        report.text_simple('Summary of dataset:')  
        report.text_simple(line_break_in_text)   
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names)}\n"
            f"HTML report: {report.report_path}\n"
            )
        report.text_simple(line_break_in_text)  
        
        # Check that the data is in the correct format and print the output in the log
        # Some functions can only take Batch as a numeric array, convert now to make a seperate array batch_numeric for these functions and 

        logger.info("Checking data format")

        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {'U', 'S', 'O'}:  # string or object (categorical)
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                # Return the batch numeric array and the unique batch name matching the numeric code
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
        else:
            raise ValueError("Batch must be a list or numpy array")
        
        # Check if save data is set to true, if so save the data as a pandas dataframe to workspace and as a csv in the save directory
        if save_data:
            logger.info("Saving results as a dictionary dataframe and .csv file")
            data_dict ={}
            data_dict["batch"]=batch
            if covariates is not None:
                for i in range(covariates.shape[1]):
                    if covariate_names is not None and i < len(covariate_names):
                        cov_name = covariate_names[i]
                    else:
                        cov_name = f'covariate_{i+1}'
                    data_dict[cov_name] = covariates[:, i]
    # Eventually save dataframe as csv in save directory after adding the outputs of each new test as new entries
            if save_data_name is None:
                save_data_name = "DiagnosticReport_InputData.csv"
        # Create batch names from unique values in batch if not provided

        # Return the number of samples per batch in the log
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

        # PLACE HOLDER SPACE FOR CHECKING BATCH SIZES AND ADVSISING MINIMUM SAMPLES PER BATCH FOR RELIABLE TESTS
        # CHECK LIT FOR MIN SAMPLES FOR RELIABLE COHENS D, MAHALANOBIS, LEVENE'S TEST, VARIANCE RATIO TEST, KS TEST
        # ADVISE IN REPORT IF ANY BATCH HAS LESS THAN MINIMUM SAMPLES, CAUTION WHEN RESULTS MAY NOT BE RELIABLE
        # PROVIDE GUIDANCE ON CORRECT APPROACH TO PCA (EG: depening on sample size and well defined effects)
        ##############################################################################################



        ##############################################################################################

        # Begin the reporting of diagnostic tests
        logger.info("Beginning diagnostic tests")
        # Additive tests first
        report.text_simple(" The order of tests is as follows: Additive tests, Multiplicative tests, Tests of distribution")
        report.text_simple(" Additive tests assess differences in means between batches, Multiplicative tests assess differences in variances between batches, and Distribution tests assess overall distributional differences between batches")
        
        report.text_simple("" \
        "Additive tests: Cohens D test for mean differences, Mahalanobis distance test for multivariate mean differences" \
        " "\
        "\nMultiplicative tests: Levene's test for variance differences, Variance ratio test between each unique batch pair" \
        " "\
        "\nBoth additive and multiplicative tests: PCA visualization of data colored by batch, PCA correlation with batch and covariates" \
        " "\
        "\nTests of distribution: Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        # Report the sample size needed for each test to be reliable

        report.text_simple(line_break_in_text)

        report.text_simple(" THIS IS A PLACE HOLDER, REPLACE WITH POWER ANALYSIS TO BASE GUIDELINES ON"
        " Note on sample sizes needed for reliable test results: " \
        "\nCohen's D test: minimum 20 samples per batch" \
        "\nMahalanobis distance test: minimum 30 samples per batch" \
        "\nLevene's test: minimum 15 samples per batch" \
        "\nVariance ratio test: minimum 15 samples per batch" \
        "\nPCA: minimum 30 samples overall, more samples needed for reliable clustering diagnostics" \
        "\nTwo-sample Kolmogorov-Smirnov test: minimum 10 samples per batch" \
        "\nIf any batch has less than the minimum samples needed, please interpret results with caution as they may not be reliable")

        report.text_simple(line_break_in_text)   
        logger.info("Additive tests:")

        # Cohen's D test for mean differences
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch)

        # If save data is set to true, add cohens d results to the saved dataframe
        report.text_simple("Cohen's D test for mean differences completed")

        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results,pair_labels=pairlabels,rep=report)
        # Add a summary to the results of the Cohen's D test in the log
        # Report the number of features with small, medium and large effect sizes based on Cohen's D thresholds
        # Do for each pairwise batch comparison:
 
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            col_name = f'CohensD_{b1}_vs_{b2}'
            if save_data:
                data_dict[col_name] = cohens_d_pair
                print(cohens_d_pair)
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        report.log_text("Cohen's D plot added to report")

        report.text_simple(line_break_in_text)   
        # Mahalanobis distance test for multivariate mean differences
        logger.info("Doing Mahalanobis distance test for multivariate mean differences")
        mahalanobis_results = DiagnosticFunctions.MahalanobisDistance(data, batch,covariates=covariates)
        report.log_text("Mahalanobis distance test for multivariate mean differences completed")
        PlotDiagnosticResults.mahalanobis_distance_plot(mahalanobis_results,rep=report)
        report.log_text("Mahalanobis distance plot added to report")
        # Summary of the Mahalanobis heatmap in the log
        logger.info("Mahalanobis distance results summary:")
        # Create summary of pairwise distances    
        pairwise_distances = mahalanobis_results['pairwise_raw']
        logger.info("Pairwise test results")
        for (b1, b2), dist in pairwise_distances.items():
            report.text_simple(f"Mahalanobis distance between {b1} and {b2}: {dist:.4f}\n"\
                               " ")
        # Return summary of centroid distances
        logger.info("Unique batch to global centroied distance test results") 
        centroid_distances = mahalanobis_results['centroid_raw']
        for (b1, b2), dist in centroid_distances.items():
            report.text_simple(f"Mahalanobis distance of {b1} to overall centroid: {dist:.4f}\n")

        centroid_resid_distance = mahalanobis_results['centroid_resid']
        for (b1, b2), dist in centroid_resid_distance.items():
            report.text_simple(f"Mahalanobis distance of {b1} to overall centroid after residualising by covariates: {dist:.4f}\n")

        if save_data:
            # Create a new entry in the data_dict for Mahalanobis distances to overall centroid
            for b, dist in centroid_distances.items():
                col_name = f'Mahalanobis_Centroid_Batch{b}'
                data_dict[col_name] = dist
            # Create a new entry in the data_dict for Mahalanobis distances to overall centroid after residualising by covariates
            for b, dist in centroid_resid_distance.items():
                col_name = f'Mahalanobis_Centroid_Resid_Batch{b}'
                data_dict[col_name] = dist

        # End of additive tests 
        report.text_simple(line_break_in_text)   
        # Multiplicative tests 
        logger.info("Multiplicative tests:")
        # Levene's test for variance differences
        logger.info("Levene's test for variance differences")
        levene_results = DiagnosticFunctions.Levene_test(data, batch, centre='median')
        report.log_text("Levene's test for variance differences completed")
        # Commenting out the plot for Levene's test as it is not yet implemented
        #PlotDiagnosticResults.plot_Levene(levene_results,report=report)
        #report.log_text("Levene's test plot added to report")
        report.text_simple(line_break_in_text)   
        # Variance ratio test between each unique batch pair
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratio = DiagnosticFunctions.Variance_ratios(data, batch, covariates=covariates)
        report.log_text("Variance ratio test between each unique batch pair completed")
        labels = [f"Batch {b1} vs Batch {b2}" for (b1,b2) in variance_ratio.keys()]
        ratio_array = np.array(list(variance_ratio.values()))

        # Summarize variance ratios robustly
        summary_rows = []
        for (b1, b2), ratios in variance_ratio.items():
            ratios = np.array(ratios)
            log_ratios = np.log(ratios)

            # Core summary stats
            mean_log = np.mean(log_ratios)
            median_log = np.median(log_ratios)
            iqr_log = np.percentile(log_ratios, [25, 75])
            prop_higher = np.mean(log_ratios > 0)
            # Back-transform to ratio scale for interpretability
            median_ratio = np.exp(median_log)
            mean_ratio = np.exp(mean_log)
            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })
            if save_data:
                # Create a new entry in the data_dict for each pairwise batch variance ratio comparison
                col_name = f'VarianceRatio_Batch{b1}_vs_Batch{b2}'
                data_dict[col_name] = ratios
                # Add the differemt summary statistics to the data_dict as well
                data_dict[f'MedianLogVarianceRatio_Batch{b1}_vs_Batch{b2}'] = median_log
                data_dict[f'MeanLogVarianceRatio_Batch{b1}_vs_Batch{b2}'] = mean_log
                data_dict[f'IQRLowerLogVarianceRatio_Batch{b1}_vs_Batch{b2}'] = iqr_log[0]
                data_dict[f'IQRUpperLogVarianceRatio_Batch{b1}_vs_Batch{b2}'] = iqr_log[1]
                data_dict[f'PropHigherLogVarianceRatio_Batch{b1}_vs_Batch{b2}'] = prop_higher
                data_dict[f'MedianVarianceRatioExp_Batch{b1}_vs_Batch{b2}'] = median_ratio
                data_dict[f'MeanVarianceRatioExp_Batch{b1}_vs_Batch{b2}'] = mean_ratio
            
            # Log the text summary of variance ratio between batches, showing the IQR 
            # and proportion of features with higher variance in batch 1 so that not just mean is used
            logger.info(
                f"Variance ratio {b1} vs {b2}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in batch {b1}\n"
            )

        # Make a summary DataFrame for easy export or inclusion in reports
        summary_df = pd.DataFrame(summary_rows)
        # Add the summary dataframe to the saved data if save_data is true
        # Example: write to your report
        PlotDiagnosticResults.variance_ratio_plot(ratio_array,labels,rep=report)
        report.log_text("Variance ratio plot(s) added to report")

        report.text_simple(line_break_in_text)   
        # Both additive and multiplicative tests
        logger.info("Running PCA")

        if covariates is not None:
            logger.info("Covariates provided, checking variable names")
            if covariate_names is None or len(covariate_names) != covariates.shape[1]:
                logger.warning("Variable names not provided or do not match number of covariates + batch. Using default names.")
                covariate_names = ['batch'] + [f'covariate_{i+1}' for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            logger.info("No covariates provided")
            covariate_names = ['batch']
        # Add batch as first variable name in variable_names
        variable_names = ['batch'] + covariate_names
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PcaCorr(data, batch, covariates=covariates,variable_names=variable_names)

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        # Report the names of covariates ued in the PCA correlation plots and the PC1 vs PC2 plot
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")

        PlotDiagnosticResults.PC_corr_plot(score, batch, covariates=covariates, variable_names=covariate_names,PC_correlations=True,rep=report,show=False)
        
        report.log_text("PCA correlation plot added to report")

        # Check if dataset size is large enough for K-means clustering and silhouette score calculation
        n_samples = data.shape[0]
        n_clusters = len(np.unique(batch))
        if n_samples >= n_clusters + 1:
            logger.info("Dataset size sufficient for clustering diagnostics, proceeding with K-means clustering and silhouette score calculation")
            # Check number of PC's required to explain 70% of variance, label as n_pcs_for_clustering
            cumulative_variance = np.cumsum(explained_variance)
            n_pcs_for_clustering = np.searchsorted(cumulative_variance, 70) + 1  # +1 because searchsorted returns index where value should be inserted
            # If 1st PC explains more than 70% variance, set n_pcs_for_clustering to 2
            if n_pcs_for_clustering < 2:
                n_pcs_for_clustering = 2
            logger.info(f"Number of PCs to explain 70% variance: {n_pcs_for_clustering}")
            PlotDiagnosticResults.pc_clustering_diagnostics(PrincipleComponents=score,
                                                                            batch=batch,
                                                                            covariates=covariates,
                                                                            variable_names=covariate_names,
                                                                            n_pcs_for_clustering=n_pcs_for_clustering,
                                                                            n_clusters_for_kmeans=n_pcs_for_clustering-1,
                                                                            rep=report,
                                                                            random_state=0,
                                                                            show=False)
            report.log_text("Clustering diagnostics plot added to report")
        else:
            logger.warning("Dataset size insufficient for clustering diagnostics, skipping K-means clustering and silhouette score calculation")
            report.log_text("Clustering diagnostics skipped due to insufficient dataset size")
        
        report.text_simple(line_break_in_text)   

        # Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_test(data, batch, feature_names=None)
        report.log_text("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair completed")
        # Return the structure of the ks_results dictionary in the log
        logger.info("KS test results structure:")
        for key, value in ks_results.items():
            if key != 'params':
                logger.info(f"Key: {key}, Value type: {type(value)}")

        logger.info("Each key pair contains:")
        report.text_simple("""- each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2)""")

        # save ks results to data_dict if save_data is true
        if save_data:
            for key, value in ks_results.items():
                if key != 'params':
                    col_name_stat = f'KS_Stat_{key}'
                    col_name_pval = f'KS_PValue_{key}'
                    col_name_pval_fdr = f'KS_PValueFDR_{key}'
                    data_dict[col_name_stat] = value['statistic']
                    data_dict[col_name_pval] = value['p_value']
                    if value['p_value_fdr'] is not None:
                        data_dict[col_name_pval_fdr] = value['p_value_fdr']
 
        PlotDiagnosticResults.KS_plot(ks_results,rep=report)
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")
        # Finalize the report
        logger.info("Diagnostic tests completed")
        
        # Before closing the report, give final summary of results as well as report how to interpret based on effect and sample sizes
        # EG: two sample KS test badly defined sub < 50 samples per batch etc.
        logger.info(f"Report saved to: {report.report_path}")
        # Save data dictionary as csv if save_data is true
        if save_data:
            import csv 
            csv_path = os.path.join(save_dir, save_data_name)
            with open(csv_path,'w',newline='') as csvfile:
                writer = csv.DictWriter(csvfile, data_dict.keys())
                # Write header
                writer.writeheader()
                writer.writerows([data_dict])
                # Write data rows   
        return data_dict if save_data else None

     
def DiagnosticReportLongitudinal():
    # Place holder for future implementation
    return None

def PostHarmonizationDiagnosticReport(data_pre, data_post, 
                                      batch, 
                                        covariates=None, 
                                            variable_names=None, 
                                                save_dir=None, 
                                                    SaveArtifacts=False, 
                                                        rep=None, 
                                                            show=False):
    # Define a basic pipeline for pre vs post harmonization diagnostic report
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults
    from DiagnoseHarmonization.LoggingTool import StatsReporter
    from DiagnoseHarmonization.LoggingTool import set_report_path

    """
    Create a diagnostic report comparing pre- and post-harmonization datasets across batches, taking into account covariates
    when relevant.
    The different tests used are all defined in DiagnosticFunctions.py and the plots in PlotDiagnosticResults, alot of the tests will match those used
    in DiagnosticReport function.
  
    Args:  
        data_pre: Pre-harmonization data (subjects x features)
        data_post: Post-harmonization data (subjects x features)
        batch: Batch labels (1D array)
        covariates: Covariate matrix (subjects x covariates), optional
        variable_names: Covariate names (list), optional
        save_dir: Directory to save report and images, optional
        SaveArtifacts: Whether to save plots as PNG images, default False
        rep: StatsReporter object for logging, optional
        show: Whether to display plots, default False

    Returns:
        - report: a HTML file containing the outputs from each diagnostic function (from DiagnosticFunctions.py) and 
        and the corresponding plots (from PlottingFunctions.py) for both pre- and post-harmonization data
        - If SaveArtefacts is set to true, all plots are also saved in the same directory as the report
        - If show is set to true, all plots are also displayed as seperare matplotlib figure windows 
    Raises:
        - ValueError: if data_pre or data_post are not 2D arrays or batch is not a
        1D array, or if the number of samples in data_pre, data_post and batch do not match.

    Note: This function is a placeholder for future implementation and currently does not contain the full logic.

    Planned tests to include: 
    Batch effect removal:
        Conditional on batch and sample sizes (must be a large enough sample): Batch prediction accuracy using classifiers (e.g., Random Forest) pre- and post-harmonization (including test and train splits
        following guidelines from Marzi et al., 2025 showing impact of conceptual data leakage on batch prediction accuracy)
        - PCA visualization of data colored by batch pre- and post-harmonization
        - PCA correlation with batch and covariates pre- and post-harmonization
        - Mahalanobis distance test for multivariate mean differences pre- and post-harmonization
        - Ratio of variances between each unique batch pair pre- and post-harmonization
        - Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair pre- and post-harmonization

    Biological variability preservation:

        Conditional on batch and sample sizes as well as availability of biological covariates: Predictive accuracy for biological covariates using 
        regression/classification models pre- and post-harmonization (including test and train splits to avoid data leakage)

        - PCA correlation with biological covariates pre- and post-harmonization
        - Correlation of features with biological covariates pre- and post-harmonization
        - Variance explained (R^2) for biological covariates using linear models pre- and post-harmonization
        - Per-column (feature) correlation with biological covariates pre- and post-harmonization using a linear model approach

        """

    return None