"""
Autor: Rentea Robert-Constantin
Acest cod a fost inspirat din https://github.com/beaupreda/semi-global-matching, care m-a ajutat foarte mult sa inteleg
operatile oferite de numpy si cum sa le aplic pentru imagini.
"""

import cv2
import numpy as np
import argparse

P1 = 10
P2 = 120
BSIZE = (3, 3)
MAX_DISPARITY = 64


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        self.direction = direction
        self.name = name


N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self):
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E, W), (SE, NW), (S, N), (SW, NE)]


def load_images(left_name, right_name):
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, BSIZE, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, BSIZE, 0, 0)
    return left, right


def normalize(volume):
    return 255.0 * volume / MAX_DISPARITY


def compute_costs(left, right):
    height = left.shape[0]
    width = left.shape[1]
    disparity = MAX_DISPARITY

    cost_volume = np.full(shape=(height, width, disparity), fill_value=255, dtype=np.int32)

    for d in range(1, disparity):
        L = left[:, d + 1:-1].astype(np.int32)
        Lplus = left[:, d + 2:].astype(np.int32)
        Lminus = left[:, d: -2].astype(np.int32)

        R = right[:, 1:-d - 1].astype(np.int32)
        Rplus = right[:, 2:-d].astype(np.int32)
        Rminus = right[:, :-d - 2].astype(np.int32)

        Lplus = (L + Lplus) / 2
        Lminus = (L + Lminus) / 2
        Lmax = np.maximum.reduce([L, Lplus, Lminus])
        Lmin = np.minimum.reduce([L, Lplus, Lminus])

        Rplus = (R + Rplus) / 2
        Rminus = (R + Rminus) / 2
        Rmax = np.maximum.reduce([R, Rplus, Rminus])
        Rmin = np.minimum.reduce([R, Rplus, Rminus])

        dl = np.maximum.reduce([np.zeros(shape=L.shape), np.abs(L - Rmax), np.abs(Rmin - L)])
        dr = np.maximum.reduce([np.zeros(shape=R.shape), np.abs(R - Lmax), np.abs(Lmin - R)])

        cost_volume[:, d + 1:-1, d] = np.minimum.reduce([dl, dr])

    return cost_volume


def dummy():
	print("something fun")

def dummy2():
	pass

def get_indices(offset, dim, direction, height):
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(i)
            else:
                y_indices.append(i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset):
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = P1
    penalties[np.abs(disparities - disparities.T) > 1] = P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, paths):
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)
    path_id = 0
    for path in paths.effective_paths:
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name))
        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                north = cost_volume[0:height, x, :]
                south = np.flip(north, axis=0)
                main_aggregation[:, x, :] = get_path_cost(north, 1)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(south, 1), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                west = cost_volume[y, 0:width, :]
                east = np.flip(west, axis=0)
                main_aggregation[y, :, :] = get_path_cost(west, 1)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(east, 1, ), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T  # Pentru directia SW avem nevoie de diagonala secundara. np.diagonal returneaza numai diagonala principala
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

    return aggregation_volume


def select_disparity(aggregation_volume):
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


def sgm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
    parser.add_argument('--save', default=False, type=bool, help='save intermediate representations')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    save_images = args.save

    paths = Paths()

    print('\nLoading images...')
    left, right = load_images(left_name, right_name)

    print('\nStarting cost computation...')
    cost_volume = compute_costs(left, right)
    if save_images:
        left_disparity_map = np.uint8(normalize(np.argmin(cost_volume[:, :, 1:], axis=2)))
        cv2.imwrite('cost_volume.png', left_disparity_map)
    print("\nStarting aggregating cost...")
    left_aggregation_volume = aggregate_costs(cost_volume, paths)

    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume)))
    if save_images:
        cv2.imwrite('disparity_map_no_post_processing.png', left_disparity_map)

    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, BSIZE[0])
    cv2.imwrite(f'disparity_map.png', left_disparity_map)


if __name__ == '__main__':
    sgm()
