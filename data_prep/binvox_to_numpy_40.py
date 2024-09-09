import os

import numpy as np

import binvox_rw

ROOT = 'ModelNet40'
CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair','cone','cup', 'curtain','desk','door', 'dresser', 'flower_pot','glass_box', 'guitar', 'keyboard','lamp','laptop','mantel','monitor', 'night_stand', 'person','piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa','stairs', 'stool','table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe','xbox']


X = {'train': [], 'test': []}
y = {'train': [], 'test': []}
for label, cl in enumerate(CLASSES):
    for split in ['train', 'test']:
        examples_dir = os.path.join('.', ROOT, cl, split)
        for example in os.listdir(examples_dir):
                if 'binvox' in example:
                    with open(os.path.join(examples_dir, example), 'rb') as file:
                        data = np.int32(binvox_rw.read_as_3d_array(file).data)
		                #padded_data = np.pad(data, 3, 'constant')
                        X[split].append(data)
                        y[split].append(label)

np.savez_compressed('modelnet40.npz',
                    X_train=X['train'],
                    X_test=X['test'],
                    y_train=y['train'],
                    y_test=y['test'])