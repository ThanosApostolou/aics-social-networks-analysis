import numpy as np

from social_networks_analysis import utils


def test_np1():
    array = np.array([[10, 11, 12, 13],
                      [14, 15, 16, 17],
                      [18, 19, 20, 21],
                      [22, 23, 24, 25]])

    length = len(array)

    column = utils.convert_symetrical_array_to_column(array, length)
    print('column')
    print(column)

    reshaped = array.reshape((length**2,))
    print('reshaped')
    print(reshaped)

    diagonal = array.diagonal()
    print('diagonal')
    print(diagonal)

    triu = np.triu(array)
    print('triu')
    print(triu)

    triu_indices = np.triu_indices(length)
    print('triu_indices')
    print(triu_indices)

    final_column = array[np.triu_indices(length)]
    print('final_column')
    print(final_column)

# def test_np2():
#     array = np.array(range(0, 10, 1))
#     print('np', array)
