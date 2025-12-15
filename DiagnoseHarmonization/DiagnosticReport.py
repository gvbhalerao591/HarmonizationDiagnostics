def LongitudinalReport(data, batch, subject_ids, timepoints, features, covariates=None,
                       covariate_names=None,
                       save_data: bool = False,
                       save_data_name: str | None = None,
                       save_dir: str | os.PathLike | None = None,
                       report_name: str | None = None,
                       SaveArtifacts: bool = False,
                       rep= None,
                       show: bool = False,
                       timestamped_reports: bool = True):
    """
    Create a diagnostic report for dataset differences across batches in longitudinal data.

    Args: 
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        subject_ids (list or np.ndarray): Subject IDs for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
    
    Outputs:
        Generates an HTML report with diagnostic plots and statistics for longitudinal data.
        If `save_data` is True, also returns a dictionary and csv with input data and results.
        If SaveArtifacts is True, saves intermediate plots to `save_dir`.
    
    """


    # Check inputs and revert to defaults as needed 

    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "LongitudinalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialized HTML report at: {rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()  # type: ignore
        report = ctx
    else:
        report = report_ctx
        # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 125
        unique_subjects = set(subject_ids)
        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of measures: {data.shape[0]}\n"
            f"Unique subjects: {len(set(subject_ids))}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        report.text_simple(line_break_in_text)

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")
        
        # Check that covariates are an array if provided (.shape[1] throwing error with a list), convert to array if needed
        if covariates is not None:
            if isinstance(covariates, list):
                covariates = np.array(covariates)
            elif not isinstance(covariates, np.ndarray):
                raise ValueError("Covariates must be a numpy array or list if provided")
        
        # Check if there is only one covariate and convert to 2D array if that is the case (avoid shape issue in next call):
        if covariates is not None and len(covariates.shape) == 1:
            covariates = covariates.reshape(-1, 1)

        # Prepare save-data dict if requested
        if save_data:
            data_dict = {}
            data_dict["batch"] = batch
            if covariates is not None:
                for i in range(covariates.shape[1]):
                    if covariate_names is not None and i < len(covariate_names):
                        cov_name = covariate_names[i]
                    else:
                        cov_name = f"covariate_{i+1}"
                    data_dict[cov_name] = covariates[:, i]
            if save_data_name is None:
                save_data_name = "DiagnosticReport_InputData.csv"
        else:
            data_dict = None
        # Check batch, subject_ids, and data dimensions
        if not (len(batch) == len(subject_ids) == data.shape[0]):
            raise ValueError("Length of batch and subject_ids must match number of samples in data")
        if len(covariates) is not None and len(covariates) != data.shape[0]:
            raise ValueError("Number of rows in covariates must match number of samples in data")
        if len(covariate_names) is not None and len(covariate_names) != covariates.shape[1]:
            raise ValueError("Length of covariate_names must match number of columns in covariates")
        

        report.log_section("Introduction", "Longitudinal Data Diagnostic Report Introduction")
        report.text_simple(
            "This report provides diagnostic analyses for longitudinal data collected across multiple batches. "
            "Longitudinal data involves repeated measurements from the same subjects over time, which introduces "
            "additional considerations for batch effects and variability. "
            "The following diagnostics will be performed:\n"
            "Mixed effects model with a subject-specific random term to show the additive effect,\n" \
            " A batch-wise variance comparison for the scaling effect,\n" \
            " Within-subject variability (coefficient of variation, percentage difference),\n" \
            " Subject order consistency across subjects and batches (Spearman correlation),\n" \
            " Cross-subject variability and preservation of biological effects (e.g., age, diagnosis, etc.). "
    )
        report.log_section("subject_order_consistency", "Subject Order Consistency Analysis")
        logger.info("PLACEHOLDER TO TEST SECTION CREATION AND PLOTTING!")
        # Subject order consistency
        results = DiagnosticFunctions.evaluate_pairwise_spearman(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            idp_names=features,
            nPerm=1000,
            seed=0,
        )
        all_results = [("subjectconsistency", {"pairwise_spearman": results})]    

        PlotDiagnosticResults.plot_pairwise_spearman_combined(all_results, save_dir, rep=report)
        report.log_text("Subject order consistency plot added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore
    
