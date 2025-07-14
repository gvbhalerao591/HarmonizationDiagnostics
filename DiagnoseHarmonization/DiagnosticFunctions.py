import argparse
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.
def Cohens_D(Data1, Data2):
    """Calculate Cohen's d for each feature between two groups."""
    n_features = len(Data1)
    d = [0] * n_features
    for f in range(n_features):
        mean1 = Data1[f].mean()
        mean2 = Data2[f].mean()
        std1 = Data1[f].std()
        std2 = Data2[f].std()
        pooled_std = ((std1 ** 2 + std2 ** 2) / 2) ** 0.5
        d[f] = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
    return d

# PcaCorr performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
def PcaCorr(Data, batch, N_components=3):
    """Perform PCA and correlate top PCs with batch labels."""

    pca = PCA(n_components=N_components)
    score = pca.fit_transform(Data)
    explained_variance = pca.explained_variance_ratio_ * 100
    batch = batch.astype(float)
    batchPCcorr = np.array([
        pearsonr(score[:, i], batch)[0]
        for i in range(min(N_components, score.shape[1]))
    ])
    return pearsonr, explained_variance, score, batchPCcorr


def MahalanobisDistance(Data=None, batch=None, Cov=None,covariates=None):
    """Calculate Mahalanobis distance between pairs of batches in the data
    Args:
        Data (np.ndarray): Data matrix where rows are samples and columns are features.
        
        batch (np.ndarray): Batch labels for each sample.

        Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
    """

    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to compute Mahalanobis distance.")
    
    batch_means = {}
    batch_data = {}

    if covariates is not None:
        from numpy.linalg import inv, pinv
        # If covariates are provided, adjust the data accordingly
        """Regress each feature in data on the covariates and return residuals."""
        # Add intercept
        X = np.column_stack([np.ones(covariates.shape[0]), covariates])  # (N, C+1)
        beta = pinv(X) @ Data  # (C+1, X)
        predicted = X @ beta   # (N, X)
        Data = Data - predicted

        return Data


    # Calculate means for each feature in each batch
    for b in unique_batches:
        batch_data[b] = Data[batch == b]
        batch_means[b] = np.mean(batch_data[b], axis=0)

    if Cov is None:
        # Compute pooled covariance matrix from the entire dataset
        Cov = np.cov(Data, rowvar=False)

    mahalanobis_distance = {}

    # Calculate Mahalanobis distance for each pair of batches
    for i,b1 in enumerate(unique_batches):
        for b2 in unique_batches[i+1:]:
            diff = batch_means[b1] - batch_means[b2]
            inv_Cov = np.linalg.inv(Cov)
            distance = np.sqrt(np.dot(np.dot(diff, inv_Cov), diff.T))
            mahalanobis_distance[(b1, b2)] = distance
    return mahalanobis_distance












# ------------------ CLI Help Only Setup ------------------

# Help functions are set up to provide descriptions of the available functions without executing them.
def setup_help_only_parser():
    parser = argparse.ArgumentParser(
        prog='DiagnosticFunctions',
        description='Diagnostic function library (use -h with a function name to view its help).'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available functions')

    # Help entry for Cohens_D
    parser_cd = subparsers.add_parser(
        'Cohens_D',
        help='Compute Cohen\'s d for two datasets',
        description="""
        Computes Cohen's d effect size per feature.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Cohens_D.py --Data1 <data1.npy> --Data2 <data2.npy>

        Returns a list of Cohen's d values for each feature.
        Data1 and Data2 should be numpy arrays with shape (features, samples).
        Each feature's Cohen's d is calculated as (mean1 - mean2) / pooled_std,
        where pooled_std is the square root of the average of the variances of both groups

        Note: This function does not handle missing values or NaNs.
        Ensure that Data1 and Data2 are preprocessed accordingly.

        '''
    )

    # Help entry for PcaCorr
    parser_pca = subparsers.add_parser(
        'PcaCorr',
        help='Perform PCA and correlate top PCs with batch',
        description="""
        Performs PCA on data and computes correlation of top N principal components with batch variable.
        Returns Pearson correlations, explained variance, PCA scores, and PC-batch correlations.
        Optional parameter:
        --N_components (default=3): Number of PCs to analyze.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.PcaCorr --Data <data.npy> --batch <batch.npy>
        Returns:
        - Pearson correlation coefficients for each PC with the batch variable.
        - Explained variance for each PC.
        - PCA scores for each sample.
        - Correlation of the first N_components PCs with the batch variable.'''
    )
    parser_mahalanobis = subparsers.add_parser(
        'mahalanobis_distance',
        help='Calculate Mahalanobis distance between batches',
        description="""
        Calculates Mahalanobis distance between pairs of batches in the data.
        If covariates are provided, it will regress each feature on the covariates and return residuals from which the Mahalanobis distance is calculated.
        Args:
            Data (np.ndarray): Data matrix where rows are samples and columns are features.
            batch (np.ndarray): Batch labels for each sample.
            Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
            covariates (np.ndarray, optional): Covariates to regress out from the data.
        Returns:
            dict: A dictionary with Mahalanobis distances for each pair of batches.
        Raises:
            ValueError: If less than two unique batches are provided.
        Example:
            mahalanobis_distance(Data=data, batch=batch_labels, Cov=cov_matrix, covariates=covariates)
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.mahalanobis_distance --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Mahalanobis distances for each pair of batches.
        '''
    )

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()