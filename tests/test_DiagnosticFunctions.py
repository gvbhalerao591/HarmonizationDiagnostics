from DiagnoseHarmonization import DiagnosticFunctions

import numpy as np

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

def test_cohens_d():
    group = np.random.rand(10,100)
    batch = np.array([0,0,0,0,0,1,1,1,1,1])
    a,b = DiagnosticFunctions.Cohens_D(group, batch)

    assert type(a) == np.ndarray
    assert type(b) == list

def test_pca_corr():
    # Create a sample dataset
    np.random.seed(0)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.zeros(100)
    batch[20:40] = batch[20:40] + 1
    batch[40:80] = batch[40:80] + 2
    batch[80:99] = batch[80:99] + 4

    # Call the PCA correlation function
    explained_variance, score, batchPCcorr, pca_ = DiagnosticFunctions.PC_Correlations(X, batch)

    # Check the shape of the results
    assert score.shape == (100, 4)  # Score should have the same number of samples as X
    assert len(explained_variance) == 4  # Explained variance should match number of features

    # Validate that each variable has 3 correlation values (one per PC)
    for var_stats in batchPCcorr.values():
        assert len(var_stats['correlation']) == 4
        assert len(var_stats['p_value']) == 4

def test_LLM_CrossSectional():
    # Create a sample dataset
    np.random.seed(0)
    data = np.random.rand(50, 10)  # 50 samples, 10 features
    subject_ids = np.array([f'subj_{i//2}' for i in range(50)])  # 25 unique subjects
    batch = np.array([i//10 for i in range(50)])  # 5 batches
    covariate_names = ['age', 'sex']
    age = np.random.randint(20, 60, size=50)
    sex = np.random.randint(0, 2, size=50)  # Binary covariate

    covariates = np.column_stack((age, sex))    
    results_df, summary = DiagnosticFunctions.Run_LMM_cross_sectional(data, batch, covariates,
                                                                     covariate_names=covariate_names,
                                                                     min_group_n=2, var_threshold=1e-8)
    # Check the shape of rsult and summary
    assert results_df.shape == (10, 11)  # 10 features, 11 results per feature
    assert isinstance(summary, dict)
    assert 'n_features' in summary
    assert summary['n_features'] == 10
    assert 'optimizer_lbfgs_no_converge' in summary
    assert 'optimizer_bfgs_no_converge' in summary  

def test_LLM_Longitudinal():
    # Create a sample dataset
    np.random.seed(0)
    data = np.random.rand(60, 15)  # 20 samples, 15 features
    # 20 unique subjects, each with 3 measurements so combine dataset 3 times
    data = np.vstack([data, data, data])
    subject_ids = np.array([f'subj_{i//3}' for i in range(60*3)])  # 20 unique subjects
    # Check subject IDs length matches data
    assert len(subject_ids) == data.shape[0]
    batch = np.array([i//15 for i in range(60*3)])  # 3 batches

    # skip covariates for now
    # Add a batch effect to some features, 1, 4, 9, 10, 12 and 14
    affected_features = [1, 4, 9, 10, 12, 14]
    for feature in affected_features:
        for b in range(3):
            data[batch == b, feature] += b * 0.5  # Incremental effect per batch
            

    # Call the LMM Longitudinal function
    results_df, summary = DiagnosticFunctions.Run_LMM_Longitudinal(data, subject_ids, batch,
                                                                    covariate_names=None,
                                                                    min_group_n=2, var_threshold=1e-8)
    # Check the shape of result and summary
    assert results_df.shape == (15, 11)  # 15 features, 6 statistics per feature
    assert isinstance(summary, dict)
    assert 'num_features_analyzed' in summary
    assert summary['num_features_analyzed'] == 15
    assert 'succeeded_LMM' in summary
    # Return a summary of the test results, pass if prints without error:
    print("LMM Longitudinal test passed successfully.")


    
        