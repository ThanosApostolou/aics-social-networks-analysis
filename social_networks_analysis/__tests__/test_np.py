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

    sources, targets = np.triu_indices(length)
    # nodes_column_list: list[str] = [f'{sources[i]}_{targets[i]}' for i, _ in enumerate(sources)]
    nodes_column = np.empty(len(sources), dtype="U32")
    for i in range(0, len(sources)):
        nodes_column[i] = f'{sources[i]}_{targets[i]}'
    print('nodes_column')
    print(nodes_column)
    print(nodes_column.dtype)


def test_np2():
    array1 = np.array([1, 2, 3, 4])
    array2 = np.array([4, 5, 6, 7])
    array = np.column_stack((array1, array2))
    print('array', array)


def test_np3():
    array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    print('np.where(array > 1)', np.where(array > 0.5)[0])

# def test_np2():
#     array = np.array(range(0, 10, 1))
#     print('np', array)
