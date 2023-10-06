import torch

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
        self.phi = 2 * torch.pi / projections
        self.resolution = resolution if resolution else n_detectors

        # recon volume dimensions
        self.xyz = None

        if x is not None and y is not None and z is not None:
            vol_recon_dims = torch.Tensor([list(i) for i in [x, y, z]])
            # shifting origin from bottom left to centre of volume
            self.xyz = vol_recon_dims - torch.Tensor([self.n/2, self.n/2, self.sdd - self.sod]).reshape(-1, 1)

    def calculate_intercepts_from_line(self, line_params):
        """
        get all pixel intercepts and make the intercept matrix
        line parameters are taken using centre of object as origin
        """
        alpha, beta, phi = line_params[:, 0].reshape(-1, 1, 1, 1), line_params[:, 1].reshape(-1, 1, 1, 1), line_params[:, 2].reshape(-1, 1, 1, 1)

        # Make pixel grid
        # each pixel is represented by its bottom left corner coordinate
        # use resolution and object size to make the
        ps = self.ps
        n = self.resolution

        x = torch.arange(-n // 2, n // 2, 1) * ps if n % 2 == 0 else torch.arange(-n, n + 1, 2) * ps / 2
        X, Y, Z = torch.meshgrid(x, x, x)

        if self.xyz is not None:
            X, Y, Z = torch.meshgrid(*[torch.arange(*self.xyz[i]) * ps for i in range(3)])

        sod = self.sod

        def xy_from_z(z):
            return alpha + z * (sod * torch.sin(phi) - alpha) / (sod * torch.cos(phi)), beta - z * beta / (sod * torch.cos(phi))

        def yz_from_x(x):
            return beta - beta * (x - alpha) / (sod * torch.sin(phi) - alpha), -(x - alpha) * sod * torch.cos(phi) / (
                    alpha - sod * torch.sin(phi))

        def zx_from_y(y):
            return -sod * torch.cos(phi) * (y - beta) / beta, alpha - (sod * torch.sin(phi) - alpha) * (y - beta) / beta

        # Get line intercepts with 6 boundaries of each voxel
        X1, Y1 = xy_from_z(Z)
        X2, Y2 = xy_from_z(Z + ps)

        Y3, Z3 = yz_from_x(X)
        Y4, Z4 = yz_from_x(X + ps)

        Z5, X5 = zx_from_y(Y)
        Z6, X6 = zx_from_y(Y + ps)

        condition1 = (X <= X1) * (X1 < X + ps) * (Y <= Y1) * (Y1 < Y + ps)
        I1 = torch.stack([torch.where(condition1, X1, 0), torch.where(condition1, Y1, 0), torch.where(condition1, Z, 0)], 4)

        condition2 = (X <= X2) * (X2 < X + ps) * (Y <= Y2) * (Y2 < Y + ps)
        I2 = torch.stack([torch.where(condition2, X2, 0), torch.where(condition2, Y2, 0), torch.where(condition2, Z + ps, 0)], 4)

        condition3 = (Y <= Y3) * (Y3 < Y + ps) * (Z <= Z3) * (Z3 < Z + ps)
        I3 = torch.stack([torch.where(condition3, X, 0), torch.where(condition3, Y3, 0), torch.where(condition3, Z3, 0)], 4)

        condition4 = (Y <= Y4) * (Y4 < Y + ps) + (Z <= Z4) * (Z4 < Z + ps)
        I4 = torch.stack([torch.where(condition4, X + ps, 0), torch.where(condition4, Y4, 0), torch.where(condition4, Z4, 0)], 4)

        condition5 = (X <= X5) * (X5 < X + ps) * (Z <= Z5) * (Z5 < Z + ps)
        I5 = torch.stack([torch.where(condition5, X5, 0), torch.where(condition5, Y, 0), torch.where(condition5, Z5, 0)], 4)

        condition6 = (X <= X6) * (X6 < X + ps) * (Z <= Z6) * (Z6 < Z + ps)
        I6 = torch.stack([torch.where(condition6, X6, 0), torch.where(condition6, Y + ps, 0), torch.where(condition6, Z6, 0)], 4)

        # To get length of line from all these intercept coordinates
        intercept_coordinates = torch.abs(torch.abs(torch.abs(I1 - I2) - torch.abs(I3 - I4)) - torch.abs(I5 - I6))

        # now squaring will give the length
        intercept_matrix = torch.linalg.norm(intercept_coordinates, dim=4)

        # change to 1d vector
        return intercept_matrix.flatten(start_dim=1, end_dim=3)

    def generate_lines(self):
        phis = (torch.arange(self.p) * self.phi).reshape(1, -1)
        n = self.n
        p = self.ps
        x = torch.arange(-n + 1, n, 2) / 2 * p if n % 2 == 0 else torch.arange((-n + 1) // 2, (n + 1) // 2) * p
        detector_coords = torch.dstack(torch.meshgrid(x, x)).reshape(-1, 2)

        mu = self.sdd - self.sod
        lambd = self.sod
        c = torch.cos(phis)
        s = torch.sin(phis)
        a = detector_coords[:, 0:1]
        b = detector_coords[:, 1:2]
        alphas = (a*lambd + lambd * mu * s)/(a*s + mu + lambd*c**2)
        betas = b / (1 - (mu + alphas*s)/(alphas*s - lambd))

        phis = phis + torch.zeros_like(alphas)
        line_params_Tensor = torch.stack([alphas, betas, phis], 2).reshape(-1, 3)

        return line_params_Tensor

    def create_intercept_matrix_from_lines(self):
        line_params_Tensor = self.generate_lines()
        self.A = self.calculate_intercepts_from_line(line_params_Tensor)
        return self.A
