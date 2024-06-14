from scipy.spatial import KDTree
import numpy as np

class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))
    

def trilinear_interpolate(grid, pts, res, grid_type='dense'):
    """
    Perform trilinear interpolation on a 3D grid at specified points. This function supports both
    dense grid structures and hashed grid representations.

    Parameters:
        grid (torch.Tensor): The grid containing data values, either dense or indexed by hash values.
        pts (torch.Tensor): Coordinates of the points for which interpolation is desired. Points should
                            be normalized between -1 and 1.
        res (int): Resolution of the grid, assumed to be cubic (res x res x res).
        grid_type (str): Type of grid storage, 'dense' for direct storage or any other string for hashed storage.

    Returns:
        torch.Tensor: Interpolated values at the input points.
    """

    PRIMES = [1, 265443567, 805459861]

    # Resize grid
    if grid_type == 'dense':
        grid = grid.reshape(res, res, res, -1)

    # Normalize
    xs = (pts[:, 0] + 1) * 0.5 * (res - 1)
    ys = (pts[:, 1] + 1) * 0.5 * (res - 1)
    zs = (pts[:, 2] + 1) * 0.5 * (res - 1)

    # Base of voxel
    x0 = torch.floor(torch.clip(xs, 0, res - 1 - 1e-5)).long()
    y0 = torch.floor(torch.clip(ys, 0, res - 1 - 1e-5)).long()
    z0 = torch.floor(torch.clip(zs, 0, res - 1 - 1e-5)).long()

    # Other corner
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Calculate weights
    w1 = ((x1 - xs) * (y1 - ys) * (z1 - zs)).unsqueeze(1)
    w2 = ((xs - x0) * (y1 - ys) * (z1 - zs)).unsqueeze(1)
    w3 = ((x1 - xs) * (ys - y0) * (z1 - zs)).unsqueeze(1)
    w4 = ((xs - x0) * (ys - y0) * (z1 - zs)).unsqueeze(1)
    w5 = ((x1 - xs) * (y1 - ys) * (zs - z0)).unsqueeze(1)
    w6 = ((xs - x0) * (y1 - ys) * (zs - z0)).unsqueeze(1)
    w7 = ((x1 - xs) * (ys - y0) * (zs - z0)).unsqueeze(1)
    w8 = ((xs - x0) * (ys - y0) * (zs - z0)).unsqueeze(1)

    # Get values, which uses hashing function if hash case
    if grid_type == 'dense':
        v1 = grid[x0, y0, z0]
        v2 = grid[x1, y0, z0]
        v3 = grid[x0, y1, z0]
        v4 = grid[x1, y1, z0]
        v5 = grid[x0, y0, z1]
        v6 = grid[x1, y0, z1]
        v7 = grid[x0, y1, z1]
        v8 = grid[x1, y1, z1]
    else:
        id1 = (x0 * PRIMES[0] ^ y0 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id2 = (x1 * PRIMES[0] ^ y0 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id3 = (x0 * PRIMES[0] ^ y1 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id4 = (x1 * PRIMES[0] ^ y1 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id5 = (x0 * PRIMES[0] ^ y0 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id6 = (x1 * PRIMES[0] ^ y0 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id7 = (x0 * PRIMES[0] ^ y1 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id8 = (x1 * PRIMES[0] ^ y1 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]

        v1 = grid[id1]
        v2 = grid[id2]
        v3 = grid[id3]
        v4 = grid[id4]
        v5 = grid[id5]
        v6 = grid[id6]
        v7 = grid[id7]
        v8 = grid[id8]

    # Lerp
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8

    return out


class Baseline():
    def __init__(self, x, y):
        self.y = y
        self.tree = KDTree(x)

    def __call__(self, x):
        _, idx = self.tree.query(x, k=3)
        return np.sign(self.y[idx].mean(axis=1))


class DenseGrid(nn.Module):
    def __init__(self, base_lod, num_lods, feature_dimension, device='cuda'):
        """
        Initializes the DenseGrid module for handling multiple levels of detail (LOD) in a dense grid.

        Parameters:
        base_lod (int): The base level of detail, where each LOD corresponds to a grid of size (2^LOD)^3.
        num_lods (int): Number of levels of detail to generate starting from the base_lod.
        feature_dimension (int): The dimensionality of features at each point in the grid.
        device (str): Device to which the grid tensors will be allocated ('cuda' or 'cpu').
        """
        super(DenseGrid, self).__init__()

        # Define grid resolutions based on lod size args
        self.lod_sizes = [2 ** l for l in range(base_lod, base_lod + num_lods)]

        self.feature_dimension = feature_dimension
        self.device = device

        self.initialize_feature_grids()

    def initialize_feature_grids(self):
        """
        Initialize the feature grids for each level of detail as a Parameter in ParameterList.
        Each grid is initialized to have a normal distribution around mean 0 with a standard deviation of 0.01.
        """
        self.feature_grids = nn.ParameterList()

        for grid_size in self.lod_sizes:
            grid_features = nn.Parameter(torch.zeros(grid_size ** 3, self.feature_dimension, dtype=torch.float32, device=self.device))
            nn.init.normal_(grid_features, mean=0, std=0.01)
            self.feature_grids.append(grid_features)

    def forward(self, points):
        """
        Define the forward pass for interpolating features at the given points from multiple levels of detail.

        Parameters:
        points (Tensor): Tensor of size (num_points, 3) containing the coordinates of points where features are to be interpolated.

        Returns:
        Tensor: Concatenated features from all levels of detail for the input points.
        """

        interpolated_features = []
        for lod_size, grid_features in zip(self.lod_sizes, self.feature_grids):
            interpolated_feature = trilinear_interpolate(grid_features, points, lod_size, grid_type='dense')
            interpolated_features.append(interpolated_feature)

        return torch.cat(interpolated_features, dim=-1)


class HashGrid(nn.Module):
    def __init__(self, minimum_resolution, maximum_resolution, num_lods, hash_bandwidth, feature_dimension, device='cuda'):
        """
        Initializes the HashGrid module for spatial hashing at multiple levels of detail.

        Parameters:
        minimum_resolution (int): Minimum resolution size at the lowest level of detail.
        maximum_resolution (int): Maximum resolution size at the highest level of detail.
        num_lods (int): Number of levels of detail to manage.
        hash_bandwidth (int): Log base 2 of the number of buckets in the hash table.
        feature_dimension (int): The dimensionality of features at each hash grid point.
        device (str): Device to which the grid tensors will be allocated ('cuda' or 'cpu').
        """
        super(HashGrid, self).__init__()

        self.minimum_resolution = minimum_resolution
        self.maximum_resolution = maximum_resolution

        self.num_lods = num_lods
        self.feature_dimension = feature_dimension
        self.device = device
        self.hash_table_size = 2 ** hash_bandwidth

        # Calculate the base for exponential growth of resolutions across LODs
        base_growth = np.exp((np.log(self.maximum_resolution) - np.log(self.minimum_resolution)) / (self.num_lods - 1))
        self.lod_resolutions = [int(1 + np.floor(self.minimum_resolution * (base_growth ** l))) for l in range(self.num_lods)]

        self.initialize_feature_grids()

    def initialize_feature_grids(self):
        """
        Initialize the feature grids for each level of detail as a Parameter in ParameterList.
        Each grid is limited by the hash table size and initialized to have a normal distribution with a very small standard deviation.
        """

        self.feature_grids = nn.ParameterList()

        for lod_size in self.lod_resolutions:
            grid_features = nn.Parameter(
                torch.zeros(min(lod_size ** 3, self.hash_table_size), self.feature_dimension, dtype=torch.float32, device=self.device))
            nn.init.normal_(grid_features, mean=0, std=0.001)
            self.feature_grids.append(grid_features)

    def forward(self, points):
        """
        Define the forward pass for interpolating features at the given points from multiple levels of detail.

        Parameters:
        points (Tensor): Tensor of size (num_points, 3) containing the coordinates of points where features are to be interpolated.

        Returns:
        Tensor: Concatenated features from all levels of detail for the input points.
        """

        interpolated_features = []

        for lod_size, grid_features in zip(self.lod_resolutions, self.feature_grids):
            if points.dim() != 2 or points.shape[1] != 3:
              raise ValueError(f"expected points to be [num_points, 3], got: {points.shape}")

            interpolated_feature = trilinear_interpolate(grid_features, points, lod_size, grid_type='hash')
            interpolated_features.append(interpolated_feature)

        concatenated_features = torch.cat(interpolated_features, dim=-1)

        return concatenated_features


class MLP(nn.Module):
    def __init__(self, num_layers, layer_width, feature_dimension, num_lods):
        """
        Initializes a multilayer perceptron (MLP) with specified parameters.

        Parameters:
        num_layers (int): Number of layers in the MLP, excluding the input and output layers.
        layer_width (int): The number of neurons in each hidden layer.
        feature_dimension (int): The dimensionality of the input features per level of detail.
        num_lods (int): The number of different levels of detail, which influences input dimension.

        The network architecture follows this sequence: Input Layer -> (num_layers) Hidden Layers -> Output Layer.
        """
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width

        self.initialize_layers(feature_dimension, num_lods)
        self.initialize_weights()

    def initialize_layers(self, feature_dimension, num_lods):
        """
        Construct the layers of the MLP from the input to the output layer.

        Parameters:
            feature_dimension (int): Dimensionality of input features per level of detail.
            num_lods (int): Number of levels of detail, determining the input size.

        This method builds each layer and adds ReLU activations after each hidden layer.
        """
        
        self.layers = nn.ModuleList()
        input_dimension = feature_dimension * num_lods

        self.layers.append(nn.Linear(input_dimension, self.layer_width))
        self.layers.append(nn.ReLU())

        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.layer_width, self.layer_width))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(self.layer_width, 1))
        self.layers = nn.Sequential(*self.layers)

    def initialize_weights(self):
        """
        Initialize weights using the Xavier uniform initializer for better initial weights distribution.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, inputs):
        """
        Defines the forward pass of the MLP using the Sequential model defined.

        Parameters:
        inputs (Tensor): The input tensor to the MLP.

        Returns:
        Tensor: Output tensor after processing through MLP and applying sigmoid activation on the output layer.
        """
        outputs = self.layers(inputs)
        outputs = torch.sigmoid(outputs)
        return outputs



class OCC(nn.Module):
    def __init__(self, config):
        """
        Initializes the OCC model which includes either a DenseGrid or HashGrid and an MLP for processing.

        Parameters:
        config (object): Configuration object containing options like grid type, dimensions, and model parameters.
        """
        super(OCC, self).__init__()

        self.config = config
        self.initialize_model()
    
    def initialize_model(self):
        """
        Set up the grid and MLP components based on the provided configuration. This method chooses between
        a DenseGrid and a HashGrid, and initializes an MLP with parameters specified in the config.
        """

        # Initialize the appropriate grid based on the config file.
        if self.config.grid_type == 'dense':
            self.grid = DenseGrid(base_lod=self.config.base_lod, num_lods=self.config.num_lods,
                                  feature_dimension=self.config.grid_feature_dimension)

            self.mlp = MLP(num_layers=self.config.num_mlp_layers, layer_width=self.config.mlp_width,
                       feature_dimension=self.config.grid_feature_dimension, num_lods=self.config.num_lods)

        elif self.config.grid_type == 'hash':
            self.grid = HashGrid(minimum_resolution=2**self.config.base_lod,
                                 maximum_resolution=2**(self.config.base_lod + self.config.num_lods - 1),
                                 num_lods=self.config.num_lods, hash_bandwidth=13,
                                 feature_dimension=self.config.grid_feature_dimension)

            self.mlp = MLP(num_layers=self.config.num_mlp_layers, layer_width=self.config.mlp_width,
                       feature_dimension=self.config.grid_feature_dimension, num_lods=self.config.num_lods)


    def forward(self, inputs):
        """
        Defines the forward pass through the grid and MLP.

        Parameters:
        inputs (np.ndarray or Tensor): Input data, if numpy array, it will be converted to a Torch Tensor.

        Returns:
        Tensor: The output of the MLP after processing inputs through the grid.
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().cuda()

        grid_output = self.grid(inputs)
        final_output = self.mlp(grid_output)
        return final_output

