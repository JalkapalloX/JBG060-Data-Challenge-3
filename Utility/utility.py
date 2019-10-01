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


def search_posterior_indices(lst, adjacent_lst):
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


def linearize_circle(p: float):
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
