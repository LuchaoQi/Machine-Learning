import numpy as np
# from statistics import mean

def cal_dist(p1,p2):
    return sum((a-b)**2 for a,b in zip(p1,p2))**0.5

def kmeans(dataset,k,tol=1e-10,max_iter=50):

    dataset = [tuple(i) for i in dataset] # using tuple as key in dictionary
    prev_centroids = {i: [] for i in dataset[:k]}
    for n in range(1, max_iter + 1):
        for data in dataset: # step1: assign datapoint to its closest centroid
            closest_centroid = min(prev_centroids,key=lambda x:cal_dist(data,x))
            prev_centroids[closest_centroid].append(data)

        cur_centroids = []
        for centroid, data in prev_centroids.items(): # calculate new centroid using assigned datapoints
            # mean_x,mean_y = mean(c[0] for c in data),mean(c[1] for c in data)
            # cur_centroids.append((mean_x,mean_y))
            arr = np.array(data)
            cur_centroid = np.mean(arr,axis=0)
            cur_centroid = tuple(cur_centroid)
            cur_centroids.append(cur_centroid)

        # return if all new centroids are close (dist <= tol) to old centroids
        if all(cal_dist(p,q) <= tol for p,q in zip(prev_centroids.keys(),cur_centroids)):
            print(f'Optimization finished with {n} iterations')
            for k,v in prev_centroids.items():
                print(k,':',v)
            break

        # update prev_centroids with cur_centroids
        prev_centroids = {centroid:[] for centroid in cur_centroids}

    return cur_centroids
    
# A = [(random.randint(0,10),random.randint(0,10)) for _ in range(1000)]
A = [[1,1],[2,1],[4,3],[5,4],[100,10],[20,20]]
kmeans(A,2)
