from linear_algebra.gauss_elimination import *
from linear_algebra.svd import *


class CreateInterceptMatrix:
    def __init__(self, no_of_detectors, source_to_object, source_to_detector, size_of_object, no_of_rotations,
                 detector_side_length, resolution=None):
        """
        Parameters
        ----------
        no_of_detectors
            no of detectors fit in 1 side length of square area
        source_to_object
            source to the centre of object
        source_to_detector
            source to any detector's centre (all detectors should be equidistance from source)
        size_of_object
            basically the length of side of square image  (which would fit the object inside it)
        detector_side_length
            side length of square detector area

        """
        self.n = no_of_detectors
        self.sod = source_to_object
        self.sdd = source_to_detector
        self.o = size_of_object
        self.r = no_of_rotations

        # Assumption: no of rotations are for 1 revolution
        self.phi = 2 * np.pi / no_of_rotations

        # square n x n x n resolution
        resolution = resolution if resolution is not None else np.cbrt(no_of_rotations * no_of_detectors ** 2)
        self.resolution = int(resolution)

        # in radians
        self.d = detector_side_length

    def calculate_intercepts_from_line(self, line_params):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using centre of object as origin

        """
        alpha, beta, phi = line_params

        # Make pixel grid
        # each pixel is represented by its bottom left corner coordinate
        # use resolution and object size to make the
        k = self.o / self.resolution
        n = self.resolution
        x = np.arange(-n // 2, n // 2, 1) * k if n % 2 == 0 else np.arange(-n, n + 1, 2) * k / 2

        X, Y, Z = np.meshgrid(x, x, x)

        sod = self.sod

        def xy_from_z(z):
            return alpha + z * (sod * np.sin(phi) - alpha) / (sod * np.cos(phi)), beta - z * beta / (sod * np.cos(phi))

        def yz_from_x(x):
            return beta - beta * (x - alpha) / (sod * np.sin(phi) - alpha), -(x - alpha) * sod * np.cos(phi) / (
                    alpha - sod * np.sin(phi))

        def zx_from_y(y):
            return -sod * np.cos(phi) * (y - beta) / beta, alpha - (sod * np.sin(phi) - alpha) * (y - beta) / beta

        # Get line intercepts with 6 boundaries of each voxel
        X1, Y1 = xy_from_z(Z)
        X2, Y2 = xy_from_z(Z + k)

        Y3, Z3 = yz_from_x(X)
        Y4, Z4 = yz_from_x(X + k)

        Z5, X5 = zx_from_y(Y)
        Z6, X6 = zx_from_y(Y + k)

        condition1 = (X <= X1) * (X1 < X + k) * (Y <= Y1) * (Y1 < Y + k)
        I1 = np.stack([np.where(condition1, X1, 0), np.where(condition1, Y1, 0), np.where(condition1, Z, 0)], 3)

        condition2 = (X <= X2) * (X2 < X + k) * (Y <= Y2) * (Y2 < Y + k)
        I2 = np.stack([np.where(condition2, X2, 0), np.where(condition2, Y2, 0), np.where(condition2, Z + k, 0)], 3)

        condition3 = (Y <= Y3) * (Y3 < Y + k) * (Z <= Z3) * (Z3 < Z + k)
        I3 = np.stack([np.where(condition3, X, 0), np.where(condition3, Y3, 0), np.where(condition3, Z3, 0)], 3)

        condition4 = (Y <= Y4) * (Y4 < Y + k) + (Z <= Z4) * (Z4 < Z + k)
        I4 = np.stack([np.where(condition4, X + k, 0), np.where(condition4, Y4, 0), np.where(condition4, Z4, 0)], 3)

        condition5 = (X <= X5) * (X5 < X + k) * (Z <= Z5) * (Z5 < Z + k)
        I5 = np.stack([np.where(condition5, X5, 0), np.where(condition5, Y, 0), np.where(condition5, Z5, 0)], 3)

        condition6 = (X <= X6) * (X6 < X + k) * (Z <= Z6) * (Z6 < Z + k)
        I6 = np.stack([np.where(condition6, X6, 0), np.where(condition6, Y + k, 0), np.where(condition6, Z6, 0)], 3)

        # To get length of line from all these intercept coordinates
        intercept_coordinates = np.abs(np.abs(np.abs(I1 - I2) - np.abs(I3 - I4)) - np.abs(I5 - I6))

        # now squaring will give the length
        intercept_matrix = np.apply_along_axis(lambda c: np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2), 3,
                                               intercept_coordinates)

        # change to 1d vector
        return intercept_matrix.flatten()

    def generate_lines(self):
        """
        will generate parameters of all the lines passing through the object
        lines are generated such 0th axis ie. rows have readings from same detector but different rotation angles

        [[d1-r1, d1-r2, d1-r3], [d2-r1, d2-r2, d3-r3]]
        """
        phis = (np.arange(self.r) * self.phi).reshape(1, -1)
        n = self.n
        d = self.d
        x = np.arange(-n + 1, n, 2) / 2 * d / n if n % 2 == 0 else np.arange((-n + 1) // 2, (n + 1) // 2)
        detector_coords = np.dstack(np.meshgrid(x, x)).reshape(-1, 2)

        gamma = self.sdd - self.sod
        lambd = self.sod
        alphas = detector_coords[:, 0:1] * lambd * np.cos(phis) + lambd * np.sin(phis) * gamma
        betas = detector_coords[:, 1:2] * lambd * np.cos(phis)

        phis = phis + np.zeros_like(alphas)
        line_params_array = np.dstack([alphas, betas, phis]).reshape(-1, 3)

        return line_params_array

    def create_intercept_matrix_from_lines(self):
        line_params_array = self.generate_lines()
        self.A = np.apply_along_axis(self.calculate_intercepts_from_line, 1, line_params_array)
        return self.A
