import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda

def pair_dataset(data_dict):
    names=list(data_dict.keys())
    X,y=[],[]
    for i,name_i in enumerate( names):
        for name_j in names[i:]:
            X.append((data_dict[name_i],data_dict[name_j]))
            y.append(all_cat(name_i,name_j))
    X=np.array(X)
    X=[X[:,0],X[:,1]]
    return X,y

def all_cat(name_i,name_j):
    return int(name_i.get_cat()==name_j.get_cat())

def distance_layer(feats_a,feats_b):
    return Lambda(euclidean_distance)([feats_a, feats_b])

def euclidean_distance(vectors):
    (feats_a, feats_b) = vectors
    sum_squared = K.sum(K.square(feats_a-feats_b), axis=1,keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss