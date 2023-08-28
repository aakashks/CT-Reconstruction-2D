import matplotlib.pyplot as plt
import numpy as np


class CreateInterceptMatrix:
    def __init__(self, no_of_detectors, source_to_object, source_to_detector, size_of_object, no_of_rotations,
                 detector_aperture):
        self.n = no_of_detectors
        self.x = source_to_object
        self.y = source_to_detector
        self.z = size_of_object
        self.r = no_of_rotations

        # Assumption: no of rotations are for 1 revolution
        self.phi = 2 * np.pi / no_of_rotations

        resolution = np.sqrt(no_of_rotations * no_of_detectors)
        # Assumption: n * r will be  square only
        assert resolution.is_integer(), 'not square resolution'
        self.a = int(resolution)

        # aperture is detector diameter size
        self.theta = 2 * np.arctan(detector_aperture / (2 * source_to_detector))

    def pixel_intercept(self, a, d, beta):
        """
        in 1 pixel calculate the length of intercept given pixel size, and line parameters taking
        one corner of the pixel as origin
        equation of line y = tan(theta) (x-d)
        d < a   (must be ensured)
        """
        assert d < a, 'd >= a'

        if (d == 0) and beta == np.pi / 2:
            # vertical edge case
            return a

        elif beta == 0:
            # horizontal edge case
            return a

        elif d >= 0:
            # enters from base
            if beta == np.pi / 2:
                # line parallel to y axis
                return a
            elif abs((a - d) / np.tan(beta)) < a:
                return abs((a - d) / np.cos(beta))
            else:
                return abs(a / np.sin(beta))

        elif d < 0:
            # enters from left pixel
            if abs((a - d) / np.tan(beta)) < a:
                return abs(a / np.cos(beta))
            else:
                return abs((a - d / np.cos(beta)) / np.cos(beta))

        else:
            raise Exception

    def get_all_pixel_interepts_from_line(self, line_params):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using bottom left corner as origin
        """
        a = self.z
        n = self.a
        d, theta = line_params
        k = a / n
        intercept_matrix = np.zeros([n, n])
        i = 0
        p = int(d // k)
        d = d % k
        while i < n and p < n:
            intercept_matrix[i][p] = self.pixel_intercept(k, d, theta)

            # checking when to move on right pixel
            if theta == 0 or abs((k - d) / np.tan(theta)) < a:
                p += 1
                d = p * k - d

            else:
                i += 1
                d = (d + k / np.tan(theta))

        # change to 1d vector
        return intercept_matrix.flatten()

    def generate_lines(self):
        """
        will generate parameters of all the lines passing through the object
        """
        phis = np.array([i * self.phi for i in range(self.r)], ndmin=2)
        thetas = np.array(
            [(i if self.n % 2 == 1 else i / 2) * self.theta for i in range(-(self.n // 2), self.n // 2 + 1, 2)], ndmin=2)

        # distances from the centre of the object
        distances_from_center = self.x * np.sin(thetas)

        # beta is slope of line (angle from +ve x-axis)
        # generate all possible values of beta for different combinations of theta and phi
        betas = (np.pi / 2 - thetas) + phis.T

        # changing origin
        distances_from_bottom_left = (1 - 1 / np.tan(betas)) * (distances_from_center + self.z / 2)
        # merge distance and angle into a couple of parameters
        line_params_array = np.dstack([distances_from_bottom_left, betas]).reshape(-1, 2)

        return line_params_array

    def create_intercept_matrix_from_lines(self):
        line_params = self.generate_lines()
        return np.apply_along_axis(self.get_all_pixel_interepts_from_line, 1, line_params)


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

    def _calculate_inverse(self):
        """
        calc A inverse
        """
        self.A_inverse_ = np.linalg.pinv(self.A)

    def solve(self, useLibrary=False):
        """
        main function to solve the equation
        """
        if useLibrary:
            self.x = np.linalg.lstsq(self.A, self.b.reshape(-1), rcond=None)[0]

        else:
            # TODO: implement Gauss Elimination method
            pass

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
        self.img_matrix = np.flip(x_vector.reshape(self.dim), 0)

    def make_figure(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.img_matrix, cmap='bone', origin='lower')
        return fig
