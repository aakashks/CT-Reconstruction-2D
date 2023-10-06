import numpy as np

class CreateInterceptMatrix:
    def __init__(self, n_detectors, source_to_object, source_to_detector, pixel_size, projections,
                 resolution=None, x=None, y=None, z=None):
        """
        Parameters
        ----------
        n_detectors
            no of detectors fit in 1 side length of square area
        source_to_object
            source to the centre of object
        source_to_detector
            source to any detector's centre (all detectors should be equidistance from source)
        pixel_size
            size of 1 pixel unit in the grid of pixels (of both object and detector, which is an ASSUMPTION)
        resolution
            ASSUMING same resolution along all axis
        x y and z are Lists containing x_min and x_max taking bottom left of detector as origin
        """
        self.n = n_detectors
        self.sod = source_to_object
        self.sdd = source_to_detector
        self.ps = pixel_size
        self.p = projections

        # Assumption: maximum angle is 2pi
        self.phi = 2 * np.pi / projections
        self.resolution = resolution if resolution else n_detectors

        # recon volume dimensions
        self.xyz = None

        if x is not None and y is not None and z is not None:
            vol_recon_dims = np.array([list(i) for i in [x, y, z]])
            # shifting origin from bottom left to centre of volume
            self.xyz = vol_recon_dims - np.array([self.n/2, self.n/2, self.sdd - self.sod]).reshape(-1, 1)

    def calculate_intercepts_from_line(self, line_params):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using centre of object as origin

        """
        alpha, beta, phi = line_params

        # Make pixel grid
        # each pixel is represented by its bottom left corner coordinate
        # use resolution and object size to make the
        ps = self.ps
        n = self.resolution

        x = np.arange(-n // 2, n // 2, 1) * ps if n % 2 == 0 else np.arange(-n, n + 1, 2) * ps / 2
        X, Y, Z = np.meshgrid(x, x, x)

        if self.xyz is not None:
            X, Y, Z = np.meshgrid(*[np.arange(*self.xyz[i]) * ps for i in range(3)])

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
        X2, Y2 = xy_from_z(Z + ps)

        Y3, Z3 = yz_from_x(X)
        Y4, Z4 = yz_from_x(X + ps)

        Z5, X5 = zx_from_y(Y)
        Z6, X6 = zx_from_y(Y + ps)

        condition1 = (X <= X1) * (X1 < X + ps) * (Y <= Y1) * (Y1 < Y + ps)
        I1 = np.stack([np.where(condition1, X1, 0), np.where(condition1, Y1, 0), np.where(condition1, Z, 0)], 3)

        condition2 = (X <= X2) * (X2 < X + ps) * (Y <= Y2) * (Y2 < Y + ps)
        I2 = np.stack([np.where(condition2, X2, 0), np.where(condition2, Y2, 0), np.where(condition2, Z + ps, 0)], 3)

        condition3 = (Y <= Y3) * (Y3 < Y + ps) * (Z <= Z3) * (Z3 < Z + ps)
        I3 = np.stack([np.where(condition3, X, 0), np.where(condition3, Y3, 0), np.where(condition3, Z3, 0)], 3)

        condition4 = (Y <= Y4) * (Y4 < Y + ps) + (Z <= Z4) * (Z4 < Z + ps)
        I4 = np.stack([np.where(condition4, X + ps, 0), np.where(condition4, Y4, 0), np.where(condition4, Z4, 0)], 3)

        condition5 = (X <= X5) * (X5 < X + ps) * (Z <= Z5) * (Z5 < Z + ps)
        I5 = np.stack([np.where(condition5, X5, 0), np.where(condition5, Y, 0), np.where(condition5, Z5, 0)], 3)

        condition6 = (X <= X6) * (X6 < X + ps) * (Z <= Z6) * (Z6 < Z + ps)
        I6 = np.stack([np.where(condition6, X6, 0), np.where(condition6, Y + ps, 0), np.where(condition6, Z6, 0)], 3)

        # To get length of line from all these intercept coordinates
        intercept_coordinates = np.abs(np.abs(np.abs(I1 - I2) - np.abs(I3 - I4)) - np.abs(I5 - I6))

        # now squaring will give the length
        intercept_matrix = np.apply_along_axis(lambda c: np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2), 3,
                                               intercept_coordinates)

        # change to 1d vector
        return intercept_matrix.flatten()

    def generate_lines(self):
        phis = (np.arange(self.p) * self.phi).reshape(1, -1)
        n = self.n
        p = self.ps
        x = np.arange(-n + 1, n, 2) / 2 * p if n % 2 == 0 else np.arange((-n + 1) // 2, (n + 1) // 2) * p
        detector_coords = np.dstack(np.meshgrid(x, x)).reshape(-1, 2)

        mu = self.sdd - self.sod
        lambd = self.sod
        c = np.cos(phis)
        s = np.sin(phis)
        a = detector_coords[:, 0:1]
        b = detector_coords[:, 1:2]
        alphas = (a*lambd + lambd * mu * s)/(a*s + mu + lambd*c**2)
        betas = b / (1 - (mu + alphas*s)/(alphas*s - lambd))

        phis = phis + np.zeros_like(alphas)
        line_params_array = np.stack([alphas, betas, phis], 2).reshape(-1, 3)

        return line_params_array

    def create_intercept_matrix_from_lines(self):
        line_params_array = self.generate_lines()
        self.A = np.apply_along_axis(self.calculate_intercepts_from_line, 1, line_params_array)
        return self.A
