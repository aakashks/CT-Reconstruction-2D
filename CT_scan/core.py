import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class CreateAMatrix:
    def __init__(self, no_of_detectors, source_to_object, source_to_detector, size_of_object, no_of_rotations,
                 detector_aperture):
        self.n = no_of_detectors
        self.x = source_to_object
        self.y = source_to_detector
        self.z = size_of_object
        self.r = no_of_rotations

        # Assumption: no of rotations are for 1 revolution
        self.phi = 360 / no_of_rotations

        resolution = np.sqrt(no_of_rotations * no_of_detectors)
        # Assumption: n * r will be  square only
        assert resolution.is_integer(), 'not square resolution'
        self.a = int(resolution)

        # Assumption: detector_aperture << source_to_detector
        self.theta = detector_aperture / source_to_detector

    def pixel_intercept(self, a, d, beta):
        """
        in 1 pixel calculate the length of intercept given pixel size, and line parameters taking
        one corner of the pixel as origin
        equation of line y = tan(theta) (x-d)
        d < a   (must be ensured)
        """
        assert d < a, 'd >= a'

        if (d == 0) and beta == np.pi:
            # vertical edge case
            return a

        elif beta == 0:
            # horizontal edge case
            return a

        elif d >= 0:
            # enters from base
            if beta == np.pi:
                # line parallel to y axis
                return a
            elif abs((a - d) / np.tan(beta)) <= a:
                return (a - d) / np.cos(beta)
            else:
                return a / np.sin(beta)

        elif d < 0:
            # enters from left pixel
            if abs((a - d) / np.tan(beta)) <= a:
                return a / np.cos(beta)
            else:
                return (a - d / np.cos(beta)) / np.cos(beta)

        else:
            raise Exception

    def get_all_pixel_interepts(self, a, n, d, theta):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using one corner as origin
        """
        k = a / n
        assert k.is_integer(), 'k not int'
        k = int(k)
        A = np.zeros([k, k])
        i = 0
        while i < n:
            p1 = d // k
            A[p1] = self.pixel_intercept(k, d % k, theta)

            # checking when to move on right pixel



class SolveEquation:
    """
    solve the linear equation Ax = b to find x_
    currently considering only a well determined system (not under determined or overdetermined)
    """

    def __init__(self, A, b):
        self.A = A
        self.b = b.reshape(-1, 1)
        # basic condition for matrix product
        assert A.shape[0] == b.shape[0], 'dimensions not matching'
        self.A_inverse_ = None
        self.x_ = None

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
            self.A_inverse_ = np.linalg.pinv(self.A)
            self.x_ = np.linalg.lstsq(self.A, self.b, rcond=None)

        else:
            self._calculate_inverse()
            self.x_ = self.A_inverse_ @ self.b

        return self.x_


class GenerateImage:
    """
    display the ouput/ results in image
    """

    def __init__(self, x_vector: np.ndarray, dim: Tuple[int, int]) -> None:
        """
        dim: specify the dimensions of the image. m x_ n image
        x_vector: the vector x_ in the equation. contains solved attenuation constant values
        """
        self.dim = dim
        self.x_vector = x_vector
        assert x_vector.size == dim[0] * dim[1], 'dimensions not matching'
        self.img_matrix = x_vector.reshape(dim)

    def make_figure(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.img_matrix, cmap='bone')
        return fig
