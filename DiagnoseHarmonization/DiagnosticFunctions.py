import numpy as np

def Cohens_D(group1, group2):
    n_features = len(group1)

    d = [0] * n_features  # Initialize a list to store Cohen's d value
    for f in range(n_features):
        
        # Calculate the means and standard deviations for each group
        mean1 = group1[f].mean()
        mean2 = group2[f].mean()
        std1 = group1[f].std()
        std2 = group2[f].std()

        # Calculate Cohen's d
        pooled_std = ((std1 ** 2 + std2 ** 2) / 2) ** 0.5
        d[f] = (mean1 - mean2) / pooled_std     

        # Check if the standard deviation is zero to avoid division by zero
        if pooled_std == 0:
            d[f] = 0 
    return d

group1 = np.array([1,2,3,4,5])
group2 = np.array([2,3,4,5,6])

a = Cohens_D(group1, group2)
print(a)