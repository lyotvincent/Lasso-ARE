"""
scARE: single-cell Autoencoder with REsidual connections
Refactored from scDCC_ResNet to accept arbitrary 2D matrices (sparse/dense)
Independent of AnnData, only matrix operations
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn import metrics
from .utils import ZINBLoss, MeanAct, DispAct, cluster_acc, check_matrix


class ResidualBlock(nn.Module):
    """Residual block with optional diffusion mechanism"""
    def __init__(self, in_features, out_features, activation="relu", diffusion=False, diffusion_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation
        self.diffusion_rate = diffusion_rate
        self.diffusion = diffusion
        
        # Diffusion control gate
        self.diffusion_gate = nn.Linear(in_features, out_features)
        self.diffusion_transform = nn.Linear(in_features, out_features)
        
        if self.activation == "relu":
            self.activation_function = nn.ReLU()
        elif self.activation == "sigmoid":
            self.activation_function = nn.Sigmoid()
            
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Ensure input and model are on the same device
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        residual = x
        
        # Main path
        out = self.fc1(x)
        out = self.activation_function(out)
        
        # Apply diffusion after first activation
        if self.diffusion and self.diffusion_rate > 0:
            # Internal diffusion mechanism
            diff_gate = torch.sigmoid(self.diffusion_gate(x))
            diff_info = self.diffusion_transform(x)
            diff_info = self.activation_function(diff_info)
            out = out * (1 - diff_gate) + diff_info * diff_gate
        
        out = self.fc2(out)
        out = self.activation_function(out)
        
        # Residual connection
        residual = self.shortcut(residual)
        out = out + residual
        out = self.activation_function(out)
        
        return out


def buildNetwork(layers, type, activation="relu", residual=True, diffusion=False, diffusion_rate=0.3):
    """
    Build neural network
    Args:
        layers: list of layer dimensions
        type: "encode" or "decode"
        activation: activation function type
        residual: whether to use residual connections
        diffusion: whether to use diffusion mechanism
        diffusion_rate: diffusion rate
    """
    net = []
    for i in range(1, len(layers)):
        if residual and i < len(layers)-1:
            net.append(ResidualBlock(layers[i-1], layers[i], activation, diffusion, diffusion_rate))
        else:
            net.append(nn.Linear(layers[i-1], layers[i]))
            if i < len(layers)-1:
                if activation == "relu":
                    net.append(nn.ReLU())
                elif activation == "sigmoid":
                    net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scARE(nn.Module):
    """
    Single-cell Autoencoder with Residual connections
    Accepts arbitrary 2D matrices (n_samples x n_features)
    """
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], 
                 activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., cl_weight=1., 
                 residual=True, device=torch.device("cpu")):
        """
        Args:
            input_dim: input feature dimension
            z_dim: latent space dimension
            n_clusters: number of clusters
            encodeLayer: encoder hidden layer dimensions
            decodeLayer: decoder hidden layer dimensions
            activation: activation function
            sigma: standard deviation for Gaussian noise
            alpha: parameter for soft assignment
            gamma: weight for clustering loss
            ml_weight: weight for must-link loss
            cl_weight: weight for cannot-link loss
            residual: whether to use residual connections
            device: torch device
        """
        super(scARE, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation, residual=residual)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation, residual=residual)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())
        self.residual = residual

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))  # cluster centers
        self.zinb_loss = ZINBLoss()
        self.device = device
    
    def save_model(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model state"""
        pretrained_dict = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def soft_assign(self, z):
        """Soft cluster assignment using Student's t-distribution"""
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        """Target distribution P"""
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input matrix [batch_size, input_dim]
        Returns:
            z0: latent representation (without noise)
            q: soft cluster assignment
            _mean: ZINB mean parameter
            _disp: ZINB dispersion parameter
            _pi: ZINB zero-inflation parameter
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        # Encoding with noise (for reconstruction)
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        # Encoding without noise (for clustering)
        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi
    
    def encodeBatch(self, X, batch_size=256):
        """
        Encode data in batches
        Args:
            X: input matrix [n_samples, n_features]
            batch_size: batch size
        Returns:
            latent representation [n_samples, z_dim]
        """
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * num / batch_size))

        with torch.no_grad():
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                if xbatch.device != self.device:
                    xbatch = xbatch.to(self.device, non_blocking=True)
                z, _, _, _, _ = self.forward(xbatch)
                encoded.append(z.detach())

        self.train()
        return torch.cat(encoded, dim=0)

    def complete_decode(self, z, batch_size=256):
        """
        Complete decoding: from latent representation to full gene expression
        Args:
            z: latent representation [n_samples, z_dim]
            batch_size: batch size
        Returns:
            mean expression [n_samples, input_dim]
        """
        self.eval()
        decoded = []
        num = z.shape[0]
        num_batch = int(math.ceil(1.0 * num / batch_size))

        with torch.no_grad():
            for batch_idx in range(num_batch):
                zbatch = z[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                if zbatch.device != self.device:
                    zbatch = zbatch.to(self.device, non_blocking=True)
                h = self.decoder(zbatch)
                mean = self._dec_mean(h)
                decoded.append(mean.detach())
        
        self.train()
        return torch.cat(decoded, dim=0)

    def decodeBatch(self, z, batch_size=256):
        """
        Decode latent representation (before mean activation)
        Args:
            z: latent representation [n_samples, z_dim]
            batch_size: batch size
        Returns:
            decoded features [n_samples, decoder_output_dim]
        """
        self.eval()
        decoded = []
        num = z.shape[0]
        num_batch = int(math.ceil(1.0 * num / batch_size))

        with torch.no_grad():
            for batch_idx in range(num_batch):
                zbatch = z[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                if zbatch.device != self.device:
                    zbatch = zbatch.to(self.device, non_blocking=True)
                h = self.decoder(zbatch)
                decoded.append(h.detach())

        self.train()
        return torch.cat(decoded, dim=0)
    
    def cluster_loss(self, p, q):
        """
        Clustering loss (KL divergence)
        Args:
            p: target distribution
            q: predicted distribution
        """
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma * kldloss

    def pairwise_loss(self, p1, p2, cons_type):
        """
        Pairwise constraint loss
        Args:
            p1: distribution for first sample
            p2: distribution for second sample
            cons_type: "ML" (must-link) or "CL" (cannot-link)
        """
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return self.ml_weight * ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return self.cl_weight * cl_loss

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, 
                            ae_save=False, ae_weights='AE_weights.pth.tar'):
        """
        Pretrain autoencoder
        Args:
            X: normalized input matrix [n_samples, n_features]
            X_raw: raw count matrix [n_samples, n_features]
            size_factor: size factors [n_samples,]
            batch_size: batch size
            lr: learning rate
            epochs: number of epochs
            ae_save: whether to save autoencoder weights
            ae_weights: path to save weights
        """
        # Convert to dense if needed
        X = check_matrix(X)
        X_raw = check_matrix(X_raw)
        
        self.to(self.device)
        self.train()
        device = self.device
        
        x_tensor = torch.tensor(X, dtype=torch.float32)
        raw_counts_tensor = torch.tensor(X_raw, dtype=torch.float32)
        sf_tensor = torch.tensor(size_factor, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, raw_counts_tensor, sf_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda"
        )
        
        print("Pretraining autoencoder...")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        
        for epoch in range(epochs):
            epoch_loss = None
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device, non_blocking=True)
                x_raw_batch = x_raw_batch.to(device, non_blocking=True)
                sf_batch = sf_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_batch)
                loss = self.zinb_loss(x=x_raw_batch, mean=mean_tensor, disp=disp_tensor, 
                                     pi=pi_tensor, scale_factor=sf_batch)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            if epoch_loss is not None:
                print(f"Pretrain epoch [{epoch+1}/{epochs}], ZINB loss: {epoch_loss:.4f}")
        
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def fit(self, X, X_raw, sf, ml_ind1=np.array([]), ml_ind2=np.array([]), 
            cl_ind1=np.array([]), cl_ind2=np.array([]), ml_p=1., cl_p=1., 
            y=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, 
            user_selected_idx=None, selected_weight=2.0, save_dir=""):
        """
        Fit clustering model
        Args:
            X: normalized input matrix [n_samples, n_features]
            X_raw: raw count matrix [n_samples, n_features]
            sf: size factors [n_samples,]
            ml_ind1, ml_ind2: must-link constraint indices
            cl_ind1, cl_ind2: cannot-link constraint indices
            ml_p, cl_p: weights for pairwise constraints
            y: true labels (for evaluation)
            lr: learning rate
            batch_size: batch size
            num_epochs: number of epochs
            update_interval: interval for updating target distribution
            tol: tolerance for early stopping
            user_selected_idx: indices of user-selected samples
            selected_weight: weight for user-selected samples
            save_dir: directory to save checkpoints
        Returns:
            y_pred: predicted cluster labels
            final_acc, final_nmi, final_ari: evaluation metrics (if y is provided)
            final_epoch: final epoch number
        """
        # Convert to dense if needed
        X = check_matrix(X)
        X_raw = check_matrix(X_raw)
        
        device = self.device
        self.to(device)
        self.train()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
        sf_tensor = torch.tensor(sf, dtype=torch.float32).to(device)
        
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        # Initialize cluster centers with kmeans
        kmeans = KMeans(self.n_clusters-1 if user_selected_idx is not None else self.n_clusters, n_init=20)
        data = self.encodeBatch(X_tensor)
        
        if user_selected_idx is not None and len(user_selected_idx) > 0:
            # User-selected cells center
            selected_data = data[user_selected_idx]
            selected_center = torch.mean(selected_data, dim=0, keepdim=True)
            
            # Run kmeans on other data points
            mask = torch.ones(data.size(0), dtype=torch.bool)
            mask[user_selected_idx] = False
            other_data = data[mask]
            other_centers = kmeans.fit(other_data.cpu().numpy()).cluster_centers_
            
            # Merge cluster centers
            cluster_centers = np.vstack([other_centers, selected_center.cpu().numpy()])
            self.mu.data.copy_(torch.tensor(cluster_centers, device=device))
            
            # Assign cluster labels
            distances = torch.cdist(data, torch.tensor(cluster_centers, device=device))
            self.y_pred = torch.argmin(distances, dim=1).cpu().numpy()
            self.y_pred_last = self.y_pred
        else:
            # Standard kmeans initialization
            self.y_pred = kmeans.fit_predict(data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, device=device))
        
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print(f'Initializing k-means: ACC= {acc:.4f}, NMI= {nmi:.4f}, ARI= {ari:.4f}')
        
        num = X_tensor.shape[0]
        num_batch = int(math.ceil(1.0*X_tensor.shape[0]/batch_size))
        
        ml_ind1_tensor = torch.tensor(ml_ind1).to(device) if ml_ind1.size > 0 else torch.tensor([]).to(device)
        ml_ind2_tensor = torch.tensor(ml_ind2).to(device) if ml_ind2.size > 0 else torch.tensor([]).to(device)
        cl_ind1_tensor = torch.tensor(cl_ind1).to(device) if cl_ind1.size > 0 else torch.tensor([]).to(device)
        cl_ind2_tensor = torch.tensor(cl_ind2).to(device) if cl_ind2.size > 0 else torch.tensor([]).to(device)
        
        ml_num_batch = int(math.ceil(1.0*len(ml_ind1_tensor)/batch_size)) if len(ml_ind1_tensor) > 0 else 0
        cl_num_batch = int(math.ceil(1.0*len(cl_ind1_tensor)/batch_size)) if len(cl_ind1_tensor) > 0 else 0
        
        cl_num = len(cl_ind1_tensor)
        ml_num = len(ml_ind1_tensor)

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # Update target distribution p
                latent = self.encodeBatch(X_tensor)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).detach()

                # Evaluate clustering performance
                self.y_pred = torch.argmax(q, dim=1).cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    final_epoch = epoch
                    print(f'Clustering {epoch+1}: ACC= {acc:.4f}, NMI= {nmi:.4f}, ARI= {ari:.4f}')

                # Check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print(f'delta_label {delta_label} < tol {tol}')
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # Train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            
            for batch_idx in range(num_batch):
                xbatch = X_tensor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw_tensor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf_tensor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                
                optimizer.zero_grad()
                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(xbatch)

                cluster_loss = self.cluster_loss(pbatch, qbatch)
                recon_loss = self.zinb_loss(xrawbatch, meanbatch, dispbatch, pibatch, sfbatch)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                
                cluster_loss_val += cluster_loss.item() * len(xbatch)
                recon_loss_val += recon_loss.item() * len(xbatch)
                
            train_loss = cluster_loss_val + recon_loss_val

            if epoch % 5 == 0 or epoch == num_epochs-1:
                print(f"#Epoch {epoch + 1:3d}: Total: {train_loss / num:.4f} "
                      f"Clustering Loss: {cluster_loss_val / num:.4f} "
                      f"ZINB Loss: {recon_loss_val / num:.4f}")

            # Must-link constraints
            ml_loss = 0.0
            if ml_num > 0 and epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    batch_start = ml_batch_idx * batch_size
                    batch_end = min(ml_num, (ml_batch_idx+1)*batch_size)
                    
                    idx1 = ml_ind1_tensor[batch_start:batch_end]
                    idx2 = ml_ind2_tensor[batch_start:batch_end]
                    
                    px1 = X_tensor[idx1]
                    px2 = X_tensor[idx2]
                    pxraw1 = X_raw_tensor[idx1]
                    pxraw2 = X_raw_tensor[idx2]
                    sf1 = sf_tensor[idx1]
                    sf2 = sf_tensor[idx2]
                    
                    optimizer.zero_grad()
                    z1, q1, mean1, disp1, pi1 = self.forward(px1)
                    z2, q2, mean2, disp2, pi2 = self.forward(px2)
                    
                    loss = (ml_p*self.pairwise_loss(q1, q2, "ML") +
                           self.zinb_loss(pxraw1, mean1, disp1, pi1, sf1) + 
                           self.zinb_loss(pxraw2, mean2, disp2, pi2, sf2))
                    
                    ml_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            # Cannot-link constraints
            cl_loss = 0.0
            if cl_num > 0 and epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    batch_start = cl_batch_idx * batch_size
                    batch_end = min(cl_num, (cl_batch_idx+1)*batch_size)
                    
                    idx1 = cl_ind1_tensor[batch_start:batch_end]
                    idx2 = cl_ind2_tensor[batch_start:batch_end]
                    
                    px1 = X_tensor[idx1]
                    px2 = X_tensor[idx2]
                    
                    optimizer.zero_grad()
                    z1, q1, _, _, _ = self.forward(px1)
                    z2, q2, _, _, _ = self.forward(px2)
                    
                    loss = cl_p*self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            if ml_num_batch > 0 and cl_num_batch > 0:
                print(f"Pairwise Total: {round(ml_loss + cl_loss, 2)}, "
                      f"ML loss: {round(ml_loss, 2)}, CL loss: {round(cl_loss, 2)}")

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
