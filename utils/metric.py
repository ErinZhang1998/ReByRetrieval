import numpy as np 

def acc_topk_with_dist(labels, arg_sorted_dist, k):
    order = arg_sorted_dist[:,1:][:,:k]
    mask = labels.reshape((-1,1)) == labels[order]
    # i = np.where(mask)[0].reshape((-1,1))
    # j = np.where(mask)[1].reshape((-1,1))
    # correct = np.concatenate([i,j],axis=1)
    # perc = np.any(mask, axis=1).sum() / len(order)
    perc_k = np.sum(mask, axis=1) / k
    # np.sum(np.sum(mask, axis=1) >= (k // 2 + 1)) / len(order)
            
    return perc_k


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: # and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # if not actual:
    #     return 0.0

    return score / k #min(len(actual), k)

def mapk_list(actual, predicted, k=10):
    return [apk(a,p,k) for a,p in zip(actual, predicted)]

def mapk(actual, predicted, k=10):
    return np.mean(mapk_list(actual, predicted, k))
