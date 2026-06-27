
def do_lasso_file(filename, user_selected_list, obs_col='annotation', vis=False, vis_key='annotation', do_correct=True):
    import sys
    # sys.path.insert(0, "/home/kyotsukitankou/selected_regenerate/NKT_tmp")
    import importlib.util
    # print(importlib.util.find_spec("pairpotlpa"))

    import h5py
    import scipy.sparse
    import time
    import random
    import numpy as np
    import pandas as pd
    import scanpy as sc

    from sklearn.metrics.cluster import adjusted_rand_score
    from lassoLPA import LPARefine
    res_list,lenval=LPARefine(selected=user_selected_list,obs_col=obs_col,file=filename,do_correct=do_correct)   
    
    if vis:
        data = sc.read_h5ad(filename)

        new_labels = np.copy(data.obs[vis_key].values) if vis_key in data.obs else np.copy(data.obsm[vis_key].values) if vis_key in data.obsm else np.array(['unlabeled'] * data.shape[0])
        user_selected = np.zeros(data.shape[0], dtype=bool)
        propagated_selected = np.zeros(data.shape[0], dtype=bool) 

        for i in user_selected_list:
            new_labels[i] = str(lenval) 
            user_selected[i] = True 

        for i in range(data.shape[0]):
            if i in res_list and not user_selected[i]: 
                propagated_selected[i] = True

        import matplotlib.pyplot as plt

        data.obs['user_selected_labels'] = pd.Categorical(new_labels)  
        data.obs['user_selected'] = pd.Categorical(user_selected) 
        data.obs['propagated_selected'] = pd.Categorical(propagated_selected)

        def set_alpha_for_scatter(ax, alpha=0.5):
            for collection in ax.collections:
                collection.set_alpha(alpha)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
        labels = ['user_selected_labels', 'user_selected', 'propagated_selected']
        for i, label in enumerate(labels):
            sc.pl.umap(data, color=label, ax=axes[i], show=False, legend_loc='on data')
            set_alpha_for_scatter(axes[i], alpha=0.5) 

        plt.tight_layout()
        plt.show()
    return res_list

def do_lasso(adata, user_selected_list, obs_col='annotation', vis=False, vis_key='annotation',do_correct=True):
    import sys
    import importlib.util

    import h5py
    import scipy.sparse
    import time
    import random
    import numpy as np
    import pandas as pd
    import scanpy as sc

    from sklearn.metrics.cluster import adjusted_rand_score
    from lassoLPA import LPA_adata
    res_list=LPA_adata(adata=adata, selected=user_selected_list,obs_col=obs_col,do_correct=do_correct)   
    lenval = len(np.unique(adata.obs[obs_col].values)) + 1
    
    if vis:
        data = adata

        new_labels = np.copy(data.obs[vis_key].values) if vis_key in data.obs else np.copy(data.obsm[vis_key].values) if vis_key in data.obsm else np.array(['unlabeled'] * data.shape[0])
        user_selected = np.zeros(data.shape[0], dtype=bool)
        propagated_selected = np.zeros(data.shape[0], dtype=bool) 

        for i in user_selected_list:
            new_labels[i] = str(lenval) 
            user_selected[i] = True 

        for i in range(data.shape[0]):
            if i in res_list and not user_selected[i]: 
                propagated_selected[i] = True

        import matplotlib.pyplot as plt

        data.obs['user_selected_labels'] = pd.Categorical(new_labels)  
        data.obs['user_selected'] = pd.Categorical(user_selected) 
        data.obs['propagated_selected'] = pd.Categorical(propagated_selected)

        def set_alpha_for_scatter(ax, alpha=0.5):
            for collection in ax.collections:
                collection.set_alpha(alpha)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
        labels = ['user_selected_labels', 'user_selected', 'propagated_selected']
        for i, label in enumerate(labels):
            sc.pl.umap(data, color=label, ax=axes[i], show=False, legend_loc='on data')
            set_alpha_for_scatter(axes[i], alpha=0.5) 

        plt.tight_layout()
        plt.show()
    return res_list