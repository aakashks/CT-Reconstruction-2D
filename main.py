from core import *
from utils import *
import pandas as pd
import json


def main():
    # INPUT using experiment_geometry.json file
    with open('exp_data_1/experiment_geometry.json') as fp:
        params = json.load(fp)

    # # INPUT from user
    # no_of_detectors = int(input("Enter no_of_detectors: "))
    # source_to_object = float(input("Enter source_to_object: "))
    # source_to_detector = float(input("Enter source_to_detector: "))
    # size_of_object = float(input("Enter size_of_object: "))
    # no_of_rotations = int(input("Enter no_of_rotations: "))
    # angle_bw_detectors = float(input("Enter angle_bw_detectors: "))

    # storing detector readings in the b vector

    # # if readings has detector no and orientations
    # detector_readings = pd.read_csv('./full_readings.csv').sort_values(by=['Rotation_no', 'Detector_no'])['Reading'].to_numpy()

    # readings where 1 column is 1 rotation
    raw_data = pd.read_csv('exp_data_1/Proj_gamma_photo_peak.csv', header=None)
    detector_readings = raw_data.to_numpy().flatten()
    d = detector_readings

    A = CreateInterceptMatrix(**params).create_intercept_matrix_from_lines()

    # TEMPORARILY using libraries for solving equation
    lambdas = SolveEquation(A, d).solve(useLibrary='lstsq')

    plot_image(lambdas.reshape(int(np.sqrt(len(lambdas))), -1))
    print('A (intercept matrix):\n', A.shape)


if __name__ == '__main__':
    main()
