'''Lorentz-equivariant fields and global SU(2) invariant contractions.'''

import numpy as np
import torch
from torch import nn

DTYPE = np.float32

# Pauli matrices
SIGMA = np.array([
    [[0, 1], [1, 0]],       # sigma_1
    [[0, -1j], [1j, 0]],    # sigma_2
    [[1, 0], [0, -1]],      # sigma_3
], dtype=complex)

# Minkowski metric diagonal
ETA = np.array([1., -1., -1., -1.], dtype=DTYPE)


class TorchMLP(nn.Module):
    '''Fully connected MLP with ReLU activations.
    Args
        output_dim (int): Dimensionality of the network output.
    '''

    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        '''Forward pass.
        Args
            x (torch.Tensor): Input tensor of shape (N, 4).
        Returns
            torch.Tensor: Output tensor of shape (N, output_dim).
        '''
        return self.net(x)


class ScalarEquivariantNN1(nn.Module):
    '''Lorentz-scalar equivariant network for NN1 (Phi).

    Computes s = t^2 - x^2 - y^2 - z^2 and passes it through
    an unconstrained MLP. Output is automatically Lorentz-invariant.
    '''

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        '''Forward pass.
        Args
            x (torch.Tensor): Spacetime coordinates of shape (N, 4).
        Returns
            torch.Tensor: Scalar field output of shape (N, 4).
        '''
        s = x[:, 0:1]**2 - (x[:, 1:2]**2 + x[:, 2:3]**2 + x[:, 3:4]**2)
        return self.mlp(s)


class VectorEquivariantNN2(nn.Module):
    '''Lorentz-vector equivariant network for NN2 (W).

    Constructs W^{mu,a} = x^mu * f^a(s) where s = x_mu x^mu and
    f^a(s) are learned scalar functions. Equivariance is exact.
    '''

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        '''Forward pass.
        Args
            x (torch.Tensor): Spacetime coordinates of shape (N, 4).
        Returns
            torch.Tensor: Flattened contravariant vector output of shape (N, 12).
        '''
        s = x[:, 0:1]**2 - (x[:, 1:2]**2 + x[:, 2:3]**2 + x[:, 3:4]**2)
        f = self.mlp(s)           # (N, 3)
        W = x.unsqueeze(2) * f.unsqueeze(1)  # (N, 4, 3)
        return W.reshape(-1, 12)


def load_nn(weights_path, output_dim):
    '''Load a pretrained TorchMLP from a .pt file.
    Args
        weights_path (str): Path to the saved state dict.
        output_dim (int): Output dimension of the network.
    Returns
        TorchMLP: Loaded model in eval mode.
    '''
    model = TorchMLP(output_dim=output_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(model, X):
    '''Evaluate a PyTorch model on a numpy array.
    Args
        model (nn.Module): PyTorch model.
        X (np.ndarray): Input array of shape (N, 4).
    Returns
        np.ndarray: Output array.
    '''
    X = np.asarray(X, dtype=DTYPE)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def reshape_W(W_flat):
    '''Reshape flat NN2 output to (N, 4, 3): W^{mu,a} with mu=Lorentz, a=SU(2) adjoint.
    Args
        W_flat (np.ndarray): Flat array of shape (N, 12).
    Returns
        np.ndarray: Reshaped array of shape (N, 4, 3).
    '''
    return W_flat.reshape(-1, 4, 3)


def random_su2():
    '''Generate a random SU(2) matrix U = exp(i alpha^a sigma^a / 2).
    Returns
        np.ndarray: Complex (2, 2) SU(2) matrix.
    '''
    alpha_vec = np.random.randn(3)
    alpha = np.linalg.norm(alpha_vec)
    if alpha < 1e-12:
        return np.eye(2, dtype=complex)
    n = alpha_vec / alpha
    U = (np.cos(alpha / 2) * np.eye(2)
         + 1j * np.sin(alpha / 2) * sum(n[a] * SIGMA[a] for a in range(3)))
    return U


def rotation_matrix(axis, theta):
    '''Build a 4x4 spatial rotation matrix.
    Args
        axis (int): Rotation axis in {0, 1, 2} for x, y, z.
        theta (float): Rotation angle in radians.
    Returns
        np.ndarray: 4x4 rotation matrix.
    '''
    R = np.eye(4)
    i, j = [(2, 3), (3, 1), (1, 2)][axis]
    R[i, i] = np.cos(theta)
    R[i, j] = -np.sin(theta)
    R[j, i] = np.sin(theta)
    R[j, j] = np.cos(theta)
    return R


def boost_matrix(axis, eta):
    '''Build a 4x4 Lorentz boost matrix.
    Args
        axis (int): Boost axis in {0, 1, 2} for x, y, z.
        eta (float): Rapidity parameter.
    Returns
        np.ndarray: 4x4 boost matrix.
    '''
    L = np.eye(4)
    s = axis + 1
    L[0, 0] = np.cosh(eta)
    L[0, s] = np.sinh(eta)
    L[s, 0] = np.sinh(eta)
    L[s, s] = np.cosh(eta)
    return L


def train_manual_nn(model, target_fn, n_steps, batch_size=256, lr=1e-3, label="model", train_range=(-5.0, 5.0)):
    '''Train a manual equivariant model by sampling fresh batches each step.
    Args
        model (nn.Module): The PyTorch model to train.
        target_fn (callable): Numpy function (N, 4) -> (N, D) producing targets.
        n_steps (int): Number of training steps.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate.
        label (str): Name for logging.
        train_range (tuple): (min, max) for uniform sampling of inputs.
    Returns
        dict: Keys "step" and "loss".
    '''
    import time

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {"step": [], "loss": []}

    print(f"\nTraining {label} for {n_steps} steps ...")
    t0 = time.time()

    for step in range(n_steps):
        X_np = np.random.uniform(train_range[0], train_range[1],
                                 size=(batch_size, 4)).astype(DTYPE)
        Y_np = target_fn(X_np)

        X_t = torch.from_numpy(X_np)
        Y_t = torch.from_numpy(Y_np)

        optimiser.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, Y_t)
        loss.backward()
        optimiser.step()

        if (step + 1) % 200 == 0 or step == 0:
            elapsed = time.time() - t0
            history["step"].append(step + 1)
            history["loss"].append(loss.item())
            print(f"  step {step+1:4d}/{n_steps}  MSE={loss.item():.6f}  ({elapsed:.1f}s)")

    print(f"  Final MSE: {loss.item():.6f}")
    return history


def central_diff(predict_fn, X, mu, eps=1e-3):
    '''Central finite difference d/dx^mu of predict_fn at points X.
    Args
        predict_fn (callable): Function (N, 4) -> (N, d).
        X (np.ndarray): Spacetime points of shape (N, 4).
        mu (int): Spacetime index 0..3.
        eps (float): Finite-difference step size.
    Returns
        np.ndarray: Derivative array of shape (N, d).
    '''
    e = np.zeros(4, dtype=DTYPE)
    e[mu] = eps
    return (predict_fn(X + e) - predict_fn(X - e)) / (2.0 * eps)


def lagrangian_density(X, predict_nn1, predict_nn2, m_phi=1.0, m_w=1.0, eps=1e-3):
    '''Evaluate the Lagrangian density L(x) at each point in X.
    Args
        X (np.ndarray): Spacetime points of shape (N, 4).
        predict_nn1 (callable): NN1 prediction function (N, 4) -> (N, 4).
        predict_nn2 (callable): NN2 prediction function (N, 4) -> (N, 12).
        m_phi (float): Scalar doublet mass.
        m_w (float): Vector boson mass.
        eps (float): Finite-difference step size.
    Returns
        np.ndarray: Lagrangian density per point of shape (N,).
    '''
    N = len(X)

    phi = predict_nn1(X)

    kinetic_phi = np.zeros(N, dtype=DTYPE)
    for mu in range(4):
        dphi_mu = central_diff(predict_nn1, X, mu, eps)
        kinetic_phi += ETA[mu] * np.sum(dphi_mu ** 2, axis=1)

    mass_phi = np.sum(phi ** 2, axis=1)

    W = predict_nn2(X).reshape(N, 4, 3)

    mass_W = np.einsum("m,nma,nma->n", ETA, W, W)

    dW = np.zeros((4, N, 4, 3), dtype=DTYPE)
    for mu in range(4):
        # Network outputs are contravariant; lower the vector index in F_mu_nu.
        dW[mu] = central_diff(predict_nn2, X, mu, eps).reshape(N, 4, 3) * ETA[None, :, None]

    F = np.zeros((4, 4, N, 3), dtype=DTYPE)
    for mu in range(4):
        for nu in range(4):
            F[mu, nu] = dW[mu][:, nu, :] - dW[nu][:, mu, :]

    kinetic_W = np.zeros(N, dtype=DTYPE)
    for mu in range(4):
        for nu in range(4):
            kinetic_W += ETA[mu] * ETA[nu] * np.sum(F[mu, nu] ** 2, axis=1)
    kinetic_W *= -0.25

    L = kinetic_phi - m_phi**2 * mass_phi + kinetic_W + 0.5 * m_w**2 * mass_W
    return L


def adjoint_matrix(U):
    '''Compute the SO(3) adjoint matrix R^{ab}(U) = (1/2) tr(sigma^a U sigma^b U^dag).
    Args
        U (np.ndarray): Complex (2, 2) SU(2) matrix.
    Returns
        np.ndarray: Real (3, 3) SO(3) rotation matrix.
    '''
    R = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            R[a, b] = 0.5 * np.real(np.trace(SIGMA[a] @ U @ SIGMA[b] @ U.conj().T))
    return R.astype(DTYPE)


def lagrangian_su2_transformed(X, U, predict_nn1, predict_nn2, m_phi=1.0, m_w=1.0, eps=1e-3):
    '''Evaluate L(x) after applying global SU(2) rotation U to the field outputs.
    Args
        X (np.ndarray): Spacetime points of shape (N, 4).
        U (np.ndarray): Complex (2, 2) SU(2) matrix.
        predict_nn1 (callable): NN1 prediction function.
        predict_nn2 (callable): NN2 prediction function.
        m_phi (float): Scalar doublet mass.
        m_w (float): Vector boson mass.
        eps (float): Finite-difference step size.
    Returns
        np.ndarray: Lagrangian density per point of shape (N,).
    '''
    N = len(X)
    R_adj = adjoint_matrix(U)

    def rotate_phi(phi_real):
        '''Rotate phi output by U (act on complex doublet).'''
        z = phi_real[:, 0::2] + 1j * phi_real[:, 1::2]
        z_rot = z @ U.T
        return np.stack([z_rot.real[:, 0], z_rot.imag[:, 0],
                         z_rot.real[:, 1], z_rot.imag[:, 1]], axis=1).astype(DTYPE)

    def predict_nn1_rot(Xb):
        return rotate_phi(predict_nn1(Xb))

    def predict_nn2_rot(Xb):
        W = predict_nn2(Xb).reshape(len(Xb), 4, 3)
        W_rot = np.einsum("ab,nmb->nma", R_adj, W)
        return W_rot.reshape(len(Xb), 12)

    return lagrangian_density(X, predict_nn1_rot, predict_nn2_rot, m_phi=m_phi, m_w=m_w, eps=eps)


def compute_equivariance_errors(predict_nn1, predict_nn2, pred_nn1_m, pred_nn2_m, n_transforms=30, seed=2024):
    '''Evaluate equivariance errors for four networks under random Lorentz transforms.
    Args
        predict_nn1 (callable): Original NN1 prediction function.
        predict_nn2 (callable): Original NN2 prediction function.
        pred_nn1_m (callable): Manual NN1 prediction function.
        pred_nn2_m (callable): Manual NN2 prediction function.
        n_transforms (int): Number of random Lorentz transformations.
        seed (int): Random seed.
    Returns
        dict: Keys are network names, values are arrays of equivariance errors.
    '''
    rng_eq = np.random.default_rng(seed)
    X_eq_test = rng_eq.uniform(-3.0, 3.0, size=(300, 4)).astype(DTYPE)

    errs = {
        "NN1_orig": [], "NN1_m": [],
        "NN2_orig": [], "NN2_m": [],
    }

    for i in range(n_transforms):
        ax_r = rng_eq.integers(0, 3)
        theta = rng_eq.uniform(0, 2 * np.pi)
        ax_b = rng_eq.integers(0, 3)
        eta = rng_eq.uniform(-1.0, 1.0)
        Lambda = boost_matrix(ax_b, eta) @ rotation_matrix(ax_r, theta)

        X_t = (Lambda @ X_eq_test.T).T.astype(DTYPE)

        y1 = predict_nn1(X_eq_test); y1t = predict_nn1(X_t)
        errs["NN1_orig"].append(np.mean(np.abs(y1t - y1)))

        y1m = pred_nn1_m(X_eq_test); y1mt = pred_nn1_m(X_t)
        errs["NN1_m"].append(np.mean(np.abs(y1mt - y1m)))

        w2 = predict_nn2(X_eq_test).reshape(-1, 4, 3)
        w2t = predict_nn2(X_t).reshape(-1, 4, 3)
        errs["NN2_orig"].append(np.mean(np.abs(w2t - np.einsum("ij,njk->nik", Lambda, w2))))

        w2m = pred_nn2_m(X_eq_test).reshape(-1, 4, 3)
        w2mt = pred_nn2_m(X_t).reshape(-1, 4, 3)
        errs["NN2_m"].append(np.mean(np.abs(w2mt - np.einsum("ij,njk->nik", Lambda, w2m))))

    for k in errs:
        errs[k] = np.array(errs[k])

    return errs
