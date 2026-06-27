import numpy as np
import scanpy as sc
import pandas as pd
import random
from collections import Counter
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import h5py
import scipy.sparse
import time
import random
import numpy as np
import sys
import os
import anndata as ad
# lpa_so_path = pkg_resources.resource_filename('pairpot', 'label_propagation.cpython-310-x86_64-linux-gnu.so')
# print(lpa_so_path)
# sys.path.append(os.path.dirname(lpa_so_path))
# import label_propagation as LPA
import pairpotlpa as LPA

# def LPARefine(adata, selected, function="anndata", do_correct=True):
#     mat=1
#     if function=="anndata":
#         mat=adata.obsp['connectivities']
#         if not scipy.sparse.issparse(mat):
#             mat=scipy.sparse.csr_matrix(mat)
#     elif function=="h5adfile":
#         with h5py.File(adata,'r') as f:
#             group=f['obsp']['connectivities']

#             data=group['data'][:]
#             indices=group['indices'][:]
#             indptr=group['indptr'][:]
#             shape=(f['obsp']['connectivities'].attrs['shape'][0],f['obsp']['connectivities'].attrs['shape'][1])

#             mat=scipy.sparse.csr_matrix((data,indices,indptr),shape=shape)
#     else:
#         print('No this function, use "anndata" or "h5adfile" instead.')
#         return
#     coo=mat.tocoo()

#     rows=coo.row
#     cols=coo.col
#     data=coo.data

#     if function=="anndata":
#         obs_col = 'annotation'
#         if obs_col not in adata.obs:
#             obs_col = 'leiden-1'

#         if "codes" in adata.obs[obs_col]:
#             mat = adata.obs[obs_col]['codes'].values
#         else:
#             mat = adata.obs[obs_col].values
#     elif function=="h5adfile":
#         with h5py.File(adata, 'r') as h5file:
#             obs_group = h5file['obs']
#             obs_col = 'annotation'
#             if obs_col not in obs_group:
#                 obs_col = 'leiden-1'

#             if "codes" in obs_group[obs_col]:
#                 mat = obs_group[obs_col]['codes'][:]
#             else:
#                 mat = obs_group[obs_col][:]
#     else:
#         print('No this function, use "anndata" or "h5adfile" instead.')
#         return
#     val={}

#     for i in np.unique(mat):
#         val[i]=len(val)
#     val[len(val)] = len(val)
#     X = LPA.matCoo(mat.shape[0], mat.shape[0])
#     for i in range(len(data)):
#         X.append(rows[i], cols[i], data[i])
                
#     y_label = LPA.mat(mat.shape[0], len(val))
#     random_list=random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))
#     select_list=np.zeros(mat.shape[0])
#     y_label.setneg()
#     select_list[random_list] = 1

#     # add selected item
#     select_list[selected] = 1
#     selected_val = len(val) - 1

#     mat_list = mat.tolist()
#     for t in range(len(selected)):
#         mat_list[selected[t]]=selected_val
#     mat = pd.Categorical(mat_list)
#     for i in range(mat.shape[0]):
#         if select_list[i]:
#             y_label.editval2(i,val[mat[i]])
#     y_pred = LPA.mat(mat.shape[0], len(val))
#     y_new = LPA.mat(mat.shape[0], len(val))
#     LPA.labelPropagation(X, y_label, y_pred, y_new, 0.5,1000)
#     y_res = np.zeros(mat.shape[0])
#     if do_correct:
#         for i in range(mat.shape[0]):
#             y_res[i] = y_new.getval(i,0)
#     else:
#         for i in range(mat.shape[0]):
#             y_res[i] = y_pred.getval(i,0)
#     y_res = pd.Series(y_res)
#     y_res = y_res[y_res == selected_val]
#     return list(y_res.index)

def LPARefine(selected,  file, use_model=LabelPropagation, obs_col='annotation',do_correct=True):
    with h5py.File(file,'r') as f:
        group=f['obsp']['connectivities']

        data=group['data'][:]
        indices=group['indices'][:]
        indptr=group['indptr'][:]
        shape=(f['obsp']['connectivities'].attrs['shape'][0],f['obsp']['connectivities'].attrs['shape'][1])

        mat=scipy.sparse.csr_matrix((data,indices,indptr),shape=shape)
    
    coo=mat.tocoo()

    rows=coo.row
    cols=coo.col
    data=coo.data

    with h5py.File(file, 'r') as h5file:
        obs_group = h5file['obs']
        if obs_col not in obs_group:
            obs_col = 'annotation'

        if "codes" in obs_group[obs_col]:
            mat = obs_group[obs_col]['codes'][:]
        else:
            mat = obs_group[obs_col][:]

    val={}

    for i in np.unique(mat):
        val[i]=len(val)
    val[len(val)] = len(val)
    X = LPA.matCoo(mat.shape[0], mat.shape[0])
    for i in range(len(data)):
        X.append(rows[i], cols[i], data[i])
                
    y_label = LPA.mat(mat.shape[0], len(val))
    random_list=random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))
    select_list=np.zeros(mat.shape[0])
    y_label.setneg()
    select_list[random_list] = 1

    # add selected item
    select_list[selected] = 1
    selected_val = len(val) - 1
    mat[selected] = selected_val
    for i in range(mat.shape[0]):
        if select_list[i]:
            y_label.editval2(i,val[mat[i]])

    y_pred = LPA.mat(mat.shape[0], len(val))
    y_new = LPA.mat(mat.shape[0], len(val))
    LPA.labelPropagation(X, y_label, y_pred, y_new, 0.5,1000)
    y_res = np.zeros(mat.shape[0])
    if do_correct:
        for i in range(mat.shape[0]):
            y_res[i] = y_new.getval(i,0)
    else:
        for i in range(mat.shape[0]):
            y_res[i] = y_pred.getval(i,0)
    y_res = pd.Series(y_res)
    y_res = y_res[y_res == selected_val]
    return list(y_res.index),selected_val


import scanpy as sc
def LPA_adata(adata, selected, use_model=LabelPropagation, obs_col='annotation',do_correct=True):
    mat=adata.obsp['connectivities']
    if not scipy.sparse.issparse(mat):
        mat=scipy.sparse.csr_matrix(mat)
    coo=mat.tocoo()

    rows=coo.row
    cols=coo.col
    data=coo.data

    if obs_col in adata.obs:
        if "codes" in adata.obs[obs_col]:
            mat = adata.obs[obs_col]['codes'].values
        else:
            mat = adata.obs[obs_col].values
    else:
        print(f'Column {obs_col} not found in adata.obs.')
        return

    val={}

    for i in np.unique(mat):
        val[i]=len(val)
    val[len(val)] = len(val)
    X = LPA.matCoo(mat.shape[0], mat.shape[0])
    for i in range(len(data)):
        X.append(rows[i], cols[i], data[i])
                
    y_label = LPA.mat(mat.shape[0], len(val))
    random_list=random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))
    select_list=np.zeros(mat.shape[0])
    y_label.setneg()
    select_list[random_list] = 1

    # add selected item
    select_list[selected] = 1
    selected_val = len(val) - 1

    mat_list = mat.tolist()
    for t in range(len(selected)):
        mat_list[selected[t]]=selected_val
    mat = pd.Categorical(mat_list)
    for i in range(mat.shape[0]):
        if select_list[i]:
            y_label.editval2(i,val[mat[i]])
    y_pred = LPA.mat(mat.shape[0], len(val))
    y_new = LPA.mat(mat.shape[0], len(val))
    LPA.labelPropagation(X, y_label, y_pred, y_new, 0.5,1000)
    y_res = np.zeros(mat.shape[0])
    if do_correct:
        for i in range(mat.shape[0]):
            y_res[i] = y_new.getval(i,0)
    else:
        for i in range(mat.shape[0]):
            y_res[i] = y_pred.getval(i,0)
    y_res = pd.Series(y_res)
    y_res = y_res[y_res == selected_val]
    return list(y_res.index)