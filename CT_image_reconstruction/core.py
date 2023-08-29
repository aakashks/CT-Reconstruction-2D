import matplotlib.pyplot as plt
import numpy as np


class CreateInterceptMatrix:
    def __init__(self, no_of_detectors, source_to_object, source_to_detector, size_of_object, no_of_rotations,
                 detector_aperture, resolution):
        self.n = no_of_detectors
        self.x = source_to_object
        self.y = source_to_detector
        self.z = size_of_object  # basically the length of side of square image  (which would fit the object inside it)
        self.r = no_of_rotations

        # Assumption: no of rotations are for 1 revolution
        self.phi = 2 * np.pi / no_of_rotations

        # square n x n resolution
        resolution = resolution if resolution else np.sqrt(no_of_rotations * no_of_detectors)
        self.a = int(resolution)

        # aperture is detector diameter size
        self.theta = 2 * np.arctan(detector_aperture / (2 * source_to_detector))

    def calculate_intercepts_from_line(self, line_params):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using bottom left corner as origin
        """
        d, theta = line_params

        # Make pixel grid
        # each pixel is represented by its bottom left corner coordinate
        # use resolution and object size to make the
        x = np.linspace(0, self.z, self.a, endpoint=False)
        k = self.z / self.a
        X, Y = np.meshgrid(x, x)

        line_from_x = lambda x: np.tan(theta) * (x - d)
        line_from_y = lambda y: y / np.tan(theta) + d

        # Get line intercepts with 4 boundaries of each pixel
        Y_left = line_from_x(X)
        Y_right = line_from_x(X + k)
        X_down = line_from_y(Y)
        X_up = line_from_y(Y + k)

        Il = np.dstack([np.where(np.logical_and(Y <= Y_left, Y_left < Y + k), X, 0),
                        np.where(np.logical_and(Y <= Y_left, Y_left < Y + k), Y_left, 0)])

        Ir = np.dstack([np.where(np.logical_and(Y <= Y_right, Y_right < Y + k), X + k, 0),
                        np.where(np.logical_and(Y <= Y_right, Y_right < Y + k), Y_right, 0)])

        Id = np.dstack([np.where(np.logical_and(X <= X_down, X_down < X + k), X_down, 0),
                        np.where(np.logical_and(X <= X_down, X_down < X + k), Y, 0)])

        Iu = np.dstack([np.where(np.logical_and(X <= X_up, X_up < X + k), X_up, 0),
                        np.where(np.logical_and(X <= X_up, X_up < X + k), Y + k, 0)])

        # To get length of line from all these intercept coordinates
        # first do |x1 - x2|, |y1 - y2| for intercept to any boundary
        intercept_coordinates = np.abs(np.abs(Il - Ir) - np.abs(Id - Iu))

        # now squaring will give the length
        intercept_matrix = np.apply_along_axis(lambda c: np.sqrt(c[0] ** 2 + c[1] ** 2), 2, intercept_coordinates)

        # change to 1d vector
        return intercept_matrix.flatten()

    def generate_lines(self):
        """
        will generate parameters of all the lines passing through the object
        """
        phis = np.array([i * self.phi for i in range(self.r)], ndmin=2)
        thetas = np.array(
            [i / 2 * self.theta for i in range(-self.n + 1, self.n, 2)]
            if self.n % 2 == 0 else
            [i * self.theta for i in range(-(self.n // 2), self.n // 2 + 1)],
            ndmin=2)

        # distances from the centre of the object
        distances_from_center = self.x * np.sin(thetas)

        # beta is slope of line (angle from +ve x-axis)
        # generate all possible values of beta for different combinations of theta and phi
        betas = (np.pi / 2 - thetas) + phis.T

        # changing origin
        distances_from_bottom_left = (1 - 1 / np.tan(betas)) * (self.z / 2) + distances_from_center / np.sin(betas)
        # merge distance and angle into a couple of parameters
        line_params_array = np.dstack([distances_from_bottom_left, betas]).reshape(-1, 2)

        return line_params_array

    def create_intercept_matrix_from_lines(self):
        line_params_array = self.generate_lines()
        return np.apply_along_axis(self.calculate_intercepts_from_line, 1, line_params_array)


class SolveEquation:
    """
    solve the linear equation Ax = b to find x_
    currently considering only a well determined system (not under determined or overdetermined)
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b
        # basic condition for matrix product
        assert A.shape[0] == b.shape[0], 'dimensions not matching'
        self.A_inverse_ = None
        self.x = None

    def gauss_elimination(self, A, b):
        n = len(b)

        # augmenting A matrix with b
        aug_matrix = np.hstack((A, b.reshape(-1, 1)))

        # Forward elimination
        for i in range(n):
            # partial pivoting
            max_row = i + np.argmax(np.abs(aug_matrix[i:, i]))
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

            pivot = aug_matrix[i, i]
            assert pivot != 0, 'singular / near-singular matrix'

            aug_matrix[i, :] /= pivot

            for j in range(i + 1, n):
                factor = aug_matrix[j, i]
                aug_matrix[j, :] -= factor * aug_matrix[i, :]

        # Backward substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = aug_matrix[i, -1] - np.dot(aug_matrix[i, i + 1:n], x[i + 1:n])

        return x

    def solve(self, useLibrary=False):
        """
        main function to solve the equation
        """
        if useLibrary == 'lstsq':
            self.x = np.linalg.lstsq(self.A, self.b.reshape(-1), rcond=None)[0]

        elif useLibrary == 'pinv':
            self.A_inverse_ = np.linalg.pinv(self.A)
            self.x = self.A_inverse_ @ self.b.reshape(-1, 1)

        else:
            self.x = self.gauss_elimination(self.A, self.b)

        return self.x


class GenerateImage:
    """
    display the ouput/ results in image
    """

    def __init__(self, x_vector, dim=None):
        """
        dim: specify the dimensions of the image. m x_ n image
        x_vector: the vector x_ in the equation. contains solved attenuation constant values
        """
        n = np.sqrt(x_vector.size)
        assert n.is_integer(), 'x_vector incorrect dimensions'
        n = int(n)
        self.dim = dim if dim else [n, n]
        self.x_vector = x_vector
        self.img_matrix = x_vector.reshape(self.dim)

    def make_figure(self, ax=None):
        if not ax:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            ax = ax
        ax.imshow(self.img_matrix, cmap='bone', origin='lower')
