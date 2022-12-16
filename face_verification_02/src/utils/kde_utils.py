import pandas as pd
from scipy.stats import gaussian_kde


def get_kde(src, labels=('Y', 'N'), lbl_col='label', dst_col='distance'):
    """
    Compute kernel density estimation of positive and negative pairs
    :param src: path to dataframe
    :param labels: tuple of positive and negative labels
    :return: kde for true and false matches
    """

    df = pd.read_csv(src)
    dist_true = df[df[lbl_col] == labels[0]][dst_col]
    dist_false = df[df[lbl_col] == labels[1]][dst_col]
    kde_true = gaussian_kde(dist_true)
    kde_false = gaussian_kde(dist_false)

    return kde_true, kde_false


def compute_confidence(distance, kde_true, kde_false):
    """
    Compute confidence of a given distance
    :param distance: input distance
    :param kde_true: kernel density estimation of true matches distances
    :param kde_false: kernel density estimation of false matches distances
    :return: confidence
    """
    y_true = kde_true(distance)[0]
    y_false = kde_false(distance)[0]
    confidence = y_true / (y_true + y_false)

    print(f'y true:  {y_true:.2f}')
    print(f'y false: {y_false:.2f}')
    print(f'Confidence (TRUE): {confidence: .5f}')

    return confidence
