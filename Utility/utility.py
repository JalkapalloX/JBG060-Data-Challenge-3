import os
import pandas as pd
import numpy as np
import datetime


def linearize_circle(p: float):
    """
    Given some level p, what proportion of a circle at unit
    diameter is filled.
    """
    r = 0.5
    area = np.pi * r**2

    if p == 0.5:
        return 0.5
    elif p == 1:
        return 1
    elif p == 0:
        return 0
    elif p < 0.5:
        angle = np.arccos((r-p)/r) * 180 / np.pi * 2
        section_area = area * angle / 360

        triangle_area = (0.5-p) * np.sqrt(0.5**2-(0.5-p)**2)

        return (section_area - triangle_area) / area
    elif p > 0.5:
        return 0.5 + 0.5-linearize_circle(1-p)

    
def reset_cumsum(lst, threshold=0, count=True):
    """
    Cummulative sum with reset at any value greater than the threshold.
    Count being true means that the output will be a cummulative count (1-2-3-...)
    otherwise its a normal cummulative sum.
    """
    output = [0]
    for i in range(1, len(lst)):
        if count:
            output = output + [0 if lst[i] >= threshold else output[i-1] + 1]
        else:
            output = output + [0 if lst[i] >= threshold else output[i-1] + lst[i]]

    return pd.Series(output, index=lst.index)


def search_prior_indices(lst, adjacent_lst):
    """
    For every object in lst, this function returns the next smaller
    element in adjadent_lst.
    """
    prior = []

    iter_adjacent_lst = iter(adjacent_lst)

    adj_index = next(iter_adjacent_lst) # current non_na_index
    prior_adj_index = -1

    for i in lst:
        if adj_index > i:
            prior += [prior_adj_index]
        else:
            while adj_index < i:
                prior_adj_index = adj_index
                try:
                    adj_index = next(iter_adjacent_lst)
                except:
                    adj_index = i+1

            prior += [prior_adj_index]

    return pd.Series(prior, index=lst)


def search_posterior_indices(lst: list, adjacent_lst: list):
    """
    For every object in lst, this function returns the next bigger
    element in adjadent_lst.
    """
    posterior = []

    iter_adjacent_lst = iter(adjacent_lst)

    adj_index = next(iter_adjacent_lst) # current non_na_index
    posterior_adj_index = -1
    last_index = max(adjacent_lst)

    for i in lst:
        if adj_index > i:
            posterior += [adj_index]
        else:
            while adj_index < i:
                try:
                    adj_index = next(iter_adjacent_lst)
                except:
                    adj_index = i+1

            if adj_index > last_index:
                adj_index = -1
            posterior += [adj_index]

    return pd.Series(posterior, index=lst)
    
    
def listfold(path):
    """
    Lists only folders in directory.
    """
    folder_names = os.listdir(path)
    folder_names = [i for i in folder_names if "." not in i]
    return folder_names

flatten = lambda l: [item for sublist in l for item in sublist]

def isolate_obj(x):
    """
    Returns string object from a list of lists.
    Useful in combination with search_for()
    """
    while str not in [type(i) for i in x]:
        x = flatten(x)
        
    for i in x:
        if type(i) is str:
            return i

def search_for(x, start_path, last_instance=True):
    """
    Searches all folder for location with x in its name.
    Returned format is weird, so wrap this function with isolate_obj()
    """
    folders = listfold(start_path)
    subpaths = [start_path + "\\" + i for i in listfold(start_path)]
    
    if x not in folders:
        path = [search_for(x, i, last_instance) for i in subpaths]

    else:
        path = start_path + "\\" + x
        
        last_path_folders = listfold(path)
        if x in last_path_folders:
            path = path + "\\" + x

    return path