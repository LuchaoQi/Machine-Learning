from statistics import mean,mode

def cal_dist(p1,p2):
    # use Euclidean distance for the Minkowski metric
    return sum((a-b)**2 for a,b in zip(p1,p2))**0.5

def knn(dataset,label, n_neighbors=5,clf=False):
    # dataset in the form of [X,y] in sklearn
    dataset = sorted(dataset,key=lambda x: cal_dist(label,x))
    candidates = dataset[:n_neighbors]
    if clf:
        return mode(candidate[-1] for candidate in candidates)
    else:
        return mean(candidate[-1] for candidate in candidates)

data =[
       [65.75, 112.99],
       [71.52, 136.49],
       [69.40, 153.03],
       [68.22, 142.34],
       [67.79, 144.30],
       [68.70, 123.30],
       [69.80, 141.49],
       [70.01, 136.46],
       [67.90, 112.37],
       [66.49, 127.45],
    ]
label = [60]

print(knn(data,label,n_neighbors=5))

data = [
    [22, 1],
    [23, 1],
    [21, 1],
    [18, 1],
    [19, 1],
    [25, 0],
    [27, 0],
    [29, 0],
    [31, 0],
    [45, 0],
]
label = [33]
print(knn(data,label,3,clf=True))
