from core import *
import pandas as pd
import json


def main():
    # INPUT using experiment_geometry.json file
    with open('./experiment_geometry.json') as fp:
        params = json.load(fp)

    # # INPUT from user
    # no_of_detectors = int(input("Enter no_of_detectors: "))
    # source_to_object = float(input("Enter source_to_object: "))
    # source_to_detector = float(input("Enter source_to_detector: "))
    # size_of_object = float(input("Enter size_of_object: "))
    # no_of_rotations = int(input("Enter no_of_rotations: "))
    # detector_aperture = float(input("Enter detector_aperture: "))

    # storing detector readings in the b vector
    # TODO: -TEMPORARY- readings are directly ln(I_o/I) values
    # # only readings
    # detector_readings = pd.read_csv('./readings.csv', header=None).to_numpy()

    # if readings has detector no and orientations
    detector_readings = pd.read_csv('./full_readings.csv').sort_values(by=['Detector_no', 'Rotation_no'])['Reading'].to_numpy()

    # # TEMPORARY: inputting from user
    # readings = input('Enter space separated readings: ')
    # detector_readings = np.array(readings.split(), dtype='float')

    d = detector_readings.reshape(-1, 1)

    A = CreateInterceptMatrix(**params).create_intercept_matrix_from_lines()

    # TEMPORARILY using libraries for solving equation
    lambdas = SolveEquation(A, d).solve(useLibrary='lstsq')

    img = GenerateImage(lambdas)
    fig = img.make_figure()
    plt.show()
    print('A (intercept matrix):\n', A)
    print('\nattenuation constants: \n', img.img_matrix)


if __name__ == '__main__':
    main()
