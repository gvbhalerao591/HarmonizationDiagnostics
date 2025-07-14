from DiagnoseHarmonization import DiagnosticFunctions
import numpy as np

def test_mahalanobis_distance():
    # Create a sample dataset
    
    np.random.seed(0)
    Data = np.random.rand(100, 5)  # 100 samples, 5 features
    batch = np.random.randint(0, 3, size=100)  # 3 unique batches

    # Call the Mahalanobis distance function
    distance = DiagnosticFunctions.MahalanobisDistance(Data, batch)
    print(distance)

    # Check the type of the result
    assert isinstance(distance, dict)

    # Check that we have distances for each pair of batches
    unique_batches = np.unique(batch)
    assert len(distance) == len(unique_batches) * (len(unique_batches) - 1) / 2

    # Check that distances are non-negative
    for key in distance:
        assert distance[key] >= 0
