import numpy as np

def pca(data):
    # scale data based on mean of features
    feature_mean = np.mean(data.T,axis=1)
    data = data - feature_mean

    covMatrix = np.cov(data.T)
    eigenValues, eigenVectors = np.linalg.eig(covMatrix)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    projections = eigenVectors.T @ data.T
    projections = projections.T

    print(eigenValues)
    print(eigenVectors)
    print(projections)
    # projections[0] is the projection of data[0] on the new coordinates

x = [-1,-1,0,2,0]
y = [-2,0,0,1,1]
data = np.array([[i,j] for i,j in zip(x,y)])
pca(data)