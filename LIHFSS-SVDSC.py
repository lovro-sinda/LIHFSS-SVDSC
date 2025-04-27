import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
#from post_clustering import spectral_clustering, acc, nmi

from matplotlib.backends.backend_pdf import PdfPages

class Conv2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)

class ConvTranspose2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]

class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.ModuleList()
        for i in range(1, len(channels)):
            self.encoder.append(nn.Sequential(
                Conv2dSamePad(kernels[i - 1], 2),
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2),
                nn.ReLU(True)
            ))

        self.decoder = nn.ModuleList()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2),
                ConvTranspose2dSamePad(kernels[i], 2),
                nn.ReLU(True)
            ))

    def forward(self, x):
        encoder_outputs = [x]
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        for layer in self.decoder:
            x = layer(x)
        return x, encoder_outputs

class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)  # Coefficient shape: [n, n], x shape: [n, d]
        return y

class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample, init_C):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)
        #self.self_expression.Coefficient.data = init_C

    def forward(self, x):  # shape=[n, c, w, h]
        x_recon, encoder_outputs = self.ae(x)
        return x_recon, encoder_outputs

    def loss_fn(self, x, x_recon, encoder_outputs, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        
        # Compute self-expression loss for each encoder layer
        loss_selfExp_all = 0
        for encoder_output in encoder_outputs:
            v = encoder_output.view(self.n, -1)  # Flatten
            v_recon = self.self_expression(v)
            loss_selfExp_all += F.mse_loss(v_recon, v, reduction='sum')
        
        # Compute self-expression loss for the last encoder layer only
        v_last = encoder_outputs[-1].view(self.n, -1)  # Flatten
        v_last_recon = self.self_expression(v_last)
        loss_selfExp_last = F.mse_loss(v_last_recon, v_last, reduction='sum')

        loss_all_layers = 0 * loss_coef + 1 * loss_selfExp_all
        loss_last_layer = weight_coef * loss_coef + weight_selfExp * loss_selfExp_last

        return loss_all_layers,0
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
nmi = normalized_mutual_info_score
def pairwise_distances(x):
    """Compute the pairwise Euclidean distances between all points in x."""
    x = x.view(x.size(0), -1)
    x_square = x.pow(2).sum(dim=1, keepdim=True)
    distances = x_square + x_square.t() - 2.0 * torch.matmul(x, x.t())
    distances = F.relu(distances)  # Replace any negative values with zero
    return distances

def distance_preserving_loss(x, encoder_outputs):
    # Compute the pairwise distances in the input space
    H = pairwise_distances(x)
    
    # Initialize the loss
    loss = 0
    
    # Calculate the loss for each encoder output
    for encoder_output in encoder_outputs:
        H_v = pairwise_distances(encoder_output)
        loss += torch.sum(H * H_v)
    
    # Average over the number of outputs
    loss /= len(encoder_outputs)
    
    return loss

# Pretraining function with distance preserving loss
def pretrain_autoencoder(ae, x, epochs, lr=1e-2, device='cuda', show=100):
    ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    else:
        x = x.to(device)
        x.requires_grad = True

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_recon, encoder_outputs = ae(x)

        # Check requires_grad for encoder outputs
        #for i, output in enumerate(encoder_outputs):
            #print(f"Encoder output {i} requires_grad: {output.requires_grad}")

        # Calculate the distance preserving loss
        loss_dp = distance_preserving_loss(x, encoder_outputs)
        
        # Backward pass
        loss_dp.backward()

        # Check gradients of model parameters
        #for name, param in ae.named_parameters():
            #if param.grad is not None:
                #print(f"Gradient for {name}: {param.grad.abs().mean()}")

        optimizer.step()

        if epoch % show == 0 or epoch == epochs - 1:
            print(f'Pretrain Epoch {epoch}, Distance Preserving Loss: {loss_dp.item()}')



import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def adjacency_matrix_angular_domain(C, delta, remove_flag, dimSubspace):
    m = C.shape[0]

    C = (np.abs(C) + np.abs(C.T)) / 2

    if remove_flag:
        CT = keep_first_kth_largest(C.T, dimSubspace)
        C = CT.T

    C = (np.abs(C) + np.abs(C.T)) / 2
    
    return C

def keep_first_kth_largest(matrix, k):
    num_rows, num_cols = matrix.shape

    if k > num_cols or k <= 0:
        raise ValueError('Invalid value of k. It should be between 1 and the number of columns in the matrix.')

    sorted_indices = np.argsort(-np.abs(matrix), axis=1)
    kth_largest_values = np.take_along_axis(np.abs(matrix), sorted_indices[:, k-1:k], axis=1)
    mask = np.abs(matrix) >= kth_largest_values
    new_matrix = matrix * mask
    return new_matrix

def spectral_clustering_shifted_laplacian(CKSym, n_clusters):
    N = CKSym.shape[0]
    MAXiter = 1000
    REPlic = 20
    CKSym= adjacency_matrix_angular_domain(CKSym, delta = 0,  remove_flag=True, dimSubspace = 9)
    DN = np.diag(1.0 / np.sqrt(CKSym.sum(axis=1) + np.finfo(float).eps))
    Lap_shifted = sp.eye(N) + DN @ CKSym @ DN

    _, _, vN = svds(Lap_shifted, k=n_clusters)
    kerN = vN.T

    kerNS = normalize(kerN, norm='l2')

    kmeans = KMeans(n_clusters=n_clusters, max_iter=MAXiter, n_init=REPlic, random_state=0)
    groups = kmeans.fit_predict(kerNS)

    return groups

def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

def compute_f(T, H):
    if len(T) != len(H):
        print("Size of T:", len(T))
        print("Size of H:", len(H))
        return None, None, None

    N = len(T)
    numT = 0
    numH = 0
    numI = 0

    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT += sum(Tn)
        numH += sum(Hn)
        numI += sum(Tn & Hn)

    p = numI / numH if numH > 0 else 1
    r = numI / numT if numT > 0 else 1
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return f

import torch
import torch.nn.functional as F

import torch.nn.functional as F

import torch.nn.functional as F

def subspace_clustering_loss(Q, C, num_clusters):
    # Compute the adjacency matrix W^(1/2)
    W_half = torch.abs(C) + torch.abs(C.T)
    W_half = torch.pow(W_half, 0.5)

    # Compute the Laplacian matrix L^(1/2)
    D_half = torch.diag(torch.sum(W_half, dim=1))
    L_half = D_half - W_half

    # Convert Q (cluster labels) to one-hot encoding without detaching from the graph
    Q_one_hot = F.one_hot(Q, num_classes=num_clusters).float()

    # Compute L_Q = tr(Q^T * L_half * Q)
    L_Q = torch.trace(Q_one_hot.T @ L_half @ Q_one_hot)
    return L_Q

def train(model, x, y, epochs, lr=1e-3, weight_coef=0, weight_selfExp=0, device='cuda', alpha=0, dim_subspace=12, ro=0, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    
    for epoch in range(epochs):
        x_recon, encoder_outputs = model(x)
        
        # Phase 1: Use multi-layer self-expression loss
        if epoch < phase1_epochs:
            loss_all_layers, _ = model.loss_fn(x, x_recon, encoder_outputs, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
            loss = loss_all_layers
        else:
            # Phase 2: Use subspace clustering loss (L_Q)
            C = model.self_expression.Coefficient  # Ensure C retains gradients
            Q_labels = spectral_clustering_shifted_laplacian(C.detach().cpu().numpy(), K)  # Run spectral clustering to get Q
            Q = torch.tensor(Q_labels, dtype=torch.int64, device=device)  # Use int64 for one-hot encoding
            loss = 10*subspace_clustering_loss(Q, C, num_clusters=K)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            A = abs(C) + abs(C).T
            y_pred = spectral_clustering_shifted_laplacian(A, K)
            print(f"Epoch {epoch}, Loss: {loss.item() / y_pred.shape[0]}, ACC: {acc(y, y_pred)}, NMI: {nmi(y, y_pred)}, Fscore: {compute_f(y,y_pred)}")
    
    return C


def compute_initial_C(X, lambd):
    X = np.array(X, dtype=np.float64)
    X = X.T
    N = X.shape[1]
    XTX = X.T @ X
    D2 = np.linalg.inv(XTX + lambd * np.eye(N))
    DD = np.diag(np.diag(D2))
    Z2 = -D2 @ np.linalg.inv(DD)
    np.fill_diagonal(Z2, 0)
    return Z2

def plot_matrix(matrix, title, pdf):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    pdf.savefig()
    plt.close()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--mode', choices=['pretrain', 'train'], required=True, help='Mode: pretrain or train')
    parser.add_argument('--db', default='coil20', choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--pretrain-epochs', default=100, type=int, help='Number of epochs for autoencoder pretraining')
    parser.add_argument('--train-epochs', default=700, type=int, help='Number of epochs for full training')
    parser.add_argument('--ae-weights', help='Path to pre-trained autoencoder weights')
    parser.add_argument('--lambd', default=0, type=float, help='Regularization parameter for initial C computation')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db

    if db == 'coil100':
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)
        print(x.shape)
        print(y.shape)
        num_sample = x.shape[0]
        channels = [1, 3]
        kernels = [15]

    if args.mode == 'pretrain':
        ae = ConvAE(channels=channels, kernels=kernels)
        ae.to(device)

        pretrain_autoencoder(ae, x, epochs=args.pretrain_epochs, lr=1e-3, device=device, show=args.show_freq)
        torch.save(ae.state_dict(), os.path.join(args.save_dir, '%s-ae.pth' % args.db))
        print("Autoencoder pretraining completed.")

    elif args.mode == 'train':
        X_flat = x.reshape(x.shape[0], -1)
        X_flat = normalize(X_flat, axis=0)
        init_C = compute_initial_C(X_flat, args.lambd)
        init_C = torch.tensor(init_C, dtype=torch.float32, device=device)

        dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels, init_C=init_C)
        dscnet.to(device)

        dscnet.ae.load_state_dict(torch.load(args.ae_weights))
        print("Pre-trained autoencoder weights loaded.")

        pdf_path = os.path.join(args.save_dir, 'C_matrices.pdf')
        with PdfPages(pdf_path) as pdf:
            plot_matrix(init_C.cpu().numpy(), "Initial C Matrix", pdf)

            final_C = train(dscnet, x, y, epochs=args.train_epochs, weight_coef=0, weight_selfExp=0, alpha=0, dim_subspace=9, ro=0, show=args.show_freq, device=device)

            plot_matrix(final_C, "Final C Matrix", pdf) #1, 150

        torch.save(dscnet.state_dict(), os.path.join(args.save_dir, '%s-model.ckp' % args.db))
        print("Full training completed and C matrices saved to PDF.")
