import numpy as np
import random
from imblearn.over_sampling import ADASYN


#balance dataset by synthetically generate instances
# 1. First, inflate the uf_label group for oversampling purpose
# 2. Next, generate "n_extra" samples with "f_label"
# 3. Return expanded dataset as a whole and the extra set separately

def balance(dataset, n_extra, inflate_rate, f_label, uf_label):

    # make a duplicate copy of the input data
    dataset_transf_train = dataset.copy(deepcopy=True)

    # subsets with favorable labels and unfavorable labels
    f_dataset = dataset.subset(np.where(dataset.labels==f_label)[0].tolist())
    uf_dataset = dataset.subset(np.where(dataset.labels==uf_label)[0].tolist())

    # expand the group with uf_label for oversampling purpose
    inflated_uf_features = np.repeat(uf_dataset.features, inflate_rate, axis=0)
    sample_features = np.concatenate((f_dataset.features, inflated_uf_features))
    inflated_uf_labels = np.repeat(uf_dataset.labels, inflate_rate, axis=0)
    sample_labels = np.concatenate((f_dataset.labels, inflated_uf_labels))

    # oversampling favorable samples
    # X: inflated dataset with synthetic samples of f_label attached to the end 
    oversample = ADASYN(sampling_strategy='minority')
    X, y = oversample.fit_resample(sample_features, sample_labels)
    y = y.reshape(-1,1)

    # take samples from dataset with only favorable labels
    X = X[np.where(y==f_label)[0].tolist()]  # data with f_label + new samples
    y = y[y==f_label]

    selected = int(f_dataset.features.shape[0]+n_extra)

    X = X[:selected, :]
    y = y[:selected]
    y = y.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = X.shape[0]-f_dataset.features.shape[0]
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]

    # compose transformed dataset
    dataset_transf_train.features = np.concatenate((uf_dataset.features, X))
    dataset_transf_train.labels = np.concatenate((uf_dataset.labels, y))
    dataset_transf_train.instance_weights = np.concatenate((uf_dataset.instance_weights, f_dataset.instance_weights, new_weights))
    dataset_transf_train.protected_attributes = np.concatenate((uf_dataset.protected_attributes, f_dataset.protected_attributes, new_attributes))

    # make a duplicate copy of the input data
    dataset_extra_train = dataset.copy()

    X_ex = X[-int(n_extra):]
    y_ex = y[-int(n_extra):]
    y_ex = y_ex.reshape(-1,1)

    # set weights and protected_attributes for the newly generated samples
    inc = int(n_extra)
    new_weights = [random.choice(f_dataset.instance_weights) for _ in range(inc)]
    new_attributes = [random.choice(f_dataset.protected_attributes) for _ in range(inc)]

    # compose extra dataset
    dataset_extra_train.features = X_ex
    dataset_extra_train.labels = y_ex
    dataset_extra_train.instance_weights = new_weights
    dataset_extra_train.protected_attributes = new_attributes

    # verifying
    #print(dataset_transf_train.features.shape)
    #print(dataset_transf_train.labels.shape)
    #print(dataset_transf_train.instance_weights.shape)
    #print(dataset_transf_train.protected_attributes.shape)

    # return favor and unfavored oversampling results
    return dataset_transf_train, dataset_extra_train


