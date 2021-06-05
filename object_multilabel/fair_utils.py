# Add functions for calculating imbalance ratio and selection algo using imbalance ratio

import numpy as np 
from tqdm import tqdm

def ir_numpy(array, epsilon=1e-4):
    return np.std(array, axis=-1)/(np.mean(array, axis=-1)+epsilon)

def k_center_fair(labelArr, imageIds, k):
    """
    Args:
        points_: Unlabelled Pool
        k: number of points to select
    returns:
        selected: Selected Pool of points
    """
    budget=k
    selected_ids = []
    assert len(labelArr)==len(imageIds)
    ind1 = np.random.choice(len(imageIds))
    # while not labelArr[ind1].all():
    #     ind1=np.random.choice(len(imageIds))
        # print(ind1)

    selected_ids.append(ind1)
    

    k-=1
    pbar = tqdm(total=budget-1)
    while k!=0:
        unselected_ids=list(set(range(len(imageIds))) - set(selected_ids))
        unselected_ids=[id_ for id_ in unselected_ids if labelArr[id_].any()]
        ########
        # print("Length of unlabelled Pool: %d " % len(unselected_ids))
        ########
        selected_labels = np.sum(labelArr[selected_ids], axis=0)
        unselected_labels = labelArr[unselected_ids]
        selected_label_repeated=np.tile(selected_labels, (unselected_labels.shape[0], 1))
        new_label_arr = selected_label_repeated+unselected_labels
        ir_list=np.apply_along_axis(ir_numpy, -1, new_label_arr)        
        
        min_ind = np.argmin(ir_list)
        p3 = unselected_ids[min_ind]
        selected_ids.append(p3)
        pbar.update(1)
        k-=1
    
    pbar.close()
    selected_img_ids = [imageIds[id_] for id_ in selected_ids]

    assert len(set(selected_img_ids))==budget

    return selected_img_ids

# Pairwise Bias Amplification S_12= log(p(c=c_1| y=w, y'=w)/p(c=c_2| y=w, y'=w)) - log(p(c=c_1| y=w)/p(c=c_2| y=w))

def customMetric(predictions, num_objects, id2obj):
    bias_amplification=[]
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            bias=0.0
            epsilon=1e-5
            class1=id2obj[i]
            class2=id2obj[j]
            arr=np.stack(list(zip(*predictions))[0])
            indices_1=np.where(arr[:,i]==1)[0].tolist()
            indices_2=np.where(arr[:,j]==1)[0].tolist()

            p_gender=np.stack(list(zip(*predictions))[1])
            true_gender=np.stack(list(zip(*predictions))[2])

            pg_women=np.where(p_gender==1)[0].tolist()
            tg_women=np.where(true_gender==1)[0].tolist()
            pg_tg_intersection=list(set(pg_women).intersection(set(tg_women)))
            cat1_pg_tg=list(set(indices_1).intersection(set(pg_tg_intersection)))
            cat2_pg_tg=list(set(indices_2).intersection(set(pg_tg_intersection)))
            # print("Predicted:", len(cat1_pg_tg), len(cat2_pg_tg))
            
            cat1_tg=list(set(indices_1).intersection(set(tg_women)))
            cat2_tg=list(set(indices_2).intersection(set(tg_women)))
            # print("GT:", len(cat1_tg), len(cat2_tg))
            
            bias=np.log((len(cat1_pg_tg)+epsilon)/(len(cat2_pg_tg)+epsilon))-np.log((len(cat1_tg)+epsilon)/(len(cat2_tg)+epsilon))            
            
            
            bias_amplification.append((class1, class2, bias))
    
    # print(np.stack(list(zip(*bias_amplification))[2]))
    average_val=np.average(np.stack(list(zip(*bias_amplification))[2]))
    print "Length of BA list: %d" % len(bias_amplification)
    return bias_amplification, abs(average_val)









