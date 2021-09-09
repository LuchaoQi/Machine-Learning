import numpy as np

x = [-1,-1,0,2,0]
y = [-2,0,0,1,1]
data = np.array([x,y])





def pca(data):

    covMatrix = np.cov(data,bias=True)


    eigenValues, eigenVectors = np.linalg.eig(covMatrix)


    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    loadings = eigenVectors * np.sqrt(eigenValues)

    print('Loadings of PC1')
    print(loadings[:,0])
    print('Projections on PC1')
    print(np.dot(data.T,loadings[:,0]))
    

pca(data)