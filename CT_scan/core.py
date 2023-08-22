import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SolveEquation:
    """
    solve the linear equation Ax = b to find x_
    currently considering only a well determined system (not under determined or overdetermined)
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b.reshape(-1, 1)
        assert A.shape[0] == b.shape[0], 'dimensions not matching'
        self.A_inverse_ = None
        self.x_ = None

    def _calculate_inverse(self):
        """
        calc A inverse
        """
        pass

    def solve(self):
        """
        main function to solve the equation
        """
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
