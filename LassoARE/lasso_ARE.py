"""
Lasso-ARE: Adversarial Autoencoder with multiple user-guided constraints
Refactored from scGAN_DCC to accept arbitrary 2D matrices (sparse/dense)
Independent of AnnData, only matrix operations

Key changes:
- Accepts list of lists for user_selected_lists (multiple user inputs)
- Multiple binary classifiers (one for each user input list) instead of one multi-class classifier
- Matrix-based operations instead of AnnData dependencies
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import math
from .scARE import scARE, buildNetwork
from .utils import check_matrix


class BinaryDiscriminator(nn.Module):
    """
    Binary discriminator for one user-selected group
    Determines if a sample belongs to the user-selected group
    """
    def __init__(self, latent_dim, n_clusters, hidden_dims=[128, 64]):
        super(BinaryDiscriminator, self).__init__()
        combined_dim = latent_dim + n_clusters
        # Use ResNet structure for feature extraction
        self.feature_net = buildNetwork([combined_dim] + hidden_dims, type="encode", 
                                       activation="relu", residual=True)
        self.attention_out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, latent, cluster_probs):
        """
        Args:
            latent: latent representation [batch_size, latent_dim]
            cluster_probs: cluster assignment probabilities [batch_size, n_clusters]
        Returns:
            attention_score: probability of belonging to the user-selected group [batch_size, 1]
        """
        combined = torch.cat([latent, cluster_probs], dim=1)
        features = self.feature_net(combined)
        attention_score = torch.sigmoid(self.attention_out(features))
        return attention_score


class MultiDiscriminator(nn.Module):
    """
    Multiple binary discriminators, one for each user-selected group
    """
    def __init__(self, latent_dim, n_clusters, num_groups, hidden_dims=[128, 64]):
        """
        Args:
            latent_dim: dimension of latent space
            n_clusters: number of clusters
            num_groups: number of user-selected groups
            hidden_dims: hidden layer dimensions for each discriminator
        """
        super(MultiDiscriminator, self).__init__()
        self.num_groups = num_groups
        
        # Create one binary discriminator for each group
        self.discriminators = nn.ModuleList([
            BinaryDiscriminator(latent_dim, n_clusters, hidden_dims)
            for _ in range(num_groups)
        ])
    
    def forward(self, latent, cluster_probs):
        """
        Args:
            latent: latent representation [batch_size, latent_dim]
            cluster_probs: cluster assignment probabilities [batch_size, n_clusters]
        Returns:
            attention_scores: list of attention scores for each group
        """
        attention_scores = []
        for disc in self.discriminators:
            score = disc(latent, cluster_probs)
            attention_scores.append(score)
        return attention_scores


class LassoARE:
    """
    Lasso-guided Adversarial autoencoding with Residual connections
    Accepts multiple user-selected groups for guided clustering
    """
    def __init__(self, input_dim, z_dim=32, n_clusters=10, 
                 enc_layers=[256, 64], dec_layers=[64, 256], disc_layers=[128, 64],
                 num_user_groups=1, residual=True, device=torch.device("cpu")):
        """
        Args:
            input_dim: input feature dimension
            z_dim: latent space dimension
            n_clusters: number of clusters
            enc_layers: encoder hidden layer dimensions
            dec_layers: decoder hidden layer dimensions
            disc_layers: discriminator hidden layer dimensions
            num_user_groups: number of user-selected groups
            residual: whether to use residual connections
            device: torch device
        """
        self.device = device
        self.n_clusters = n_clusters
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.num_user_groups = num_user_groups
        
        # Generator (clustering autoencoder)
        self.generator = scARE(
            input_dim=input_dim,
            z_dim=z_dim,
            n_clusters=n_clusters,
            encodeLayer=enc_layers,
            decodeLayer=dec_layers,
            residual=residual,
            device=device
        ).to(device)
        
        # Multiple binary discriminators
        self.discriminator = MultiDiscriminator(
            latent_dim=z_dim,
            n_clusters=n_clusters,
            num_groups=num_user_groups,
            hidden_dims=disc_layers
        ).to(device)
        
    def pretrain_generator(self, X, X_raw, size_factor, batch_size=256, lr=0.001, 
                          epochs=300, save_path=None):
        """
        Pretrain the autoencoder (generator)
        Args:
            X: normalized input matrix [n_samples, n_features]
            X_raw: raw count matrix [n_samples, n_features]
            size_factor: size factors [n_samples,]
            batch_size: batch size
            lr: learning rate
            epochs: number of epochs
            save_path: path to save weights
        """
        X = check_matrix(X)
        X_raw = check_matrix(X_raw)
        
        self.generator.pretrain_autoencoder(
            X, X_raw, size_factor, 
            batch_size=batch_size, 
            lr=lr, 
            epochs=epochs, 
            ae_save=save_path is not None,
            ae_weights=save_path
        )
        
    def feature_separation_loss(self, z, batch_masks, inner_alpha=1.0, outer_alpha=0.5, margin=1.0):
        """
        Feature separation loss for multiple user-selected groups
        Encourages samples in each group to be close to their group center
        and far from other groups' centers
        
        Args:
            z: latent representations [batch_size, z_dim]
            batch_masks: list of boolean masks for each group [num_groups, batch_size]
            inner_alpha: weight for inner cohesion
            outer_alpha: weight for outer separation
            margin: margin for separation
        Returns:
            total_inner_loss: sum of inner cohesion losses
            total_outer_loss: sum of outer separation losses
        """
        total_inner_loss = torch.tensor(0.0, device=self.device)
        total_outer_loss = torch.tensor(0.0, device=self.device)
        
        group_centers = []
        
        # Calculate center for each group
        for mask in batch_masks:
            if torch.any(mask):
                selected_z = z[mask]
                center = torch.mean(selected_z, dim=0, keepdim=True)
                group_centers.append(center)
                
                # Inner loss: minimize distance within group
                if selected_z.size(0) > 1:
                    distances = torch.cdist(selected_z, center).squeeze()
                    inner_loss = inner_alpha * (distances ** 2).mean()
                    total_inner_loss += inner_loss
            else:
                group_centers.append(None)
        
        # Outer loss: maximize distance between groups
        for i, (mask_i, center_i) in enumerate(zip(batch_masks, group_centers)):
            if center_i is None or not torch.any(mask_i):
                continue
                
            # For samples not in this group, push them away from this center
            for j, mask_j in enumerate(batch_masks):
                if i == j:
                    continue
                if torch.any(mask_j):
                    other_z = z[mask_j]
                    distances = torch.cdist(other_z, center_i).squeeze()
                    # Hinge loss: encourage distance > margin
                    outer_loss = outer_alpha * torch.relu(margin - distances).mean()
                    total_outer_loss += outer_loss
        
        return total_inner_loss, total_outer_loss
    
    def consistency_loss(self, z, batch_masks):
        """
        Consistency loss: reconstruction of selected samples should preserve their latent representation
        Args:
            z: latent representations [batch_size, z_dim]
            batch_masks: list of boolean masks for each group
        Returns:
            consistency loss
        """
        total_consistency = torch.tensor(0.0, device=self.device)
        
        for mask in batch_masks:
            if not torch.any(mask):
                continue
            
            selected_z = z[mask]
            
            with torch.no_grad():
                # Full decode and re-encode
                full_decoded = self.generator.complete_decode(selected_z)
                recoded_z = self.generator.encodeBatch(full_decoded)
            
            consistency = nn.MSELoss()(recoded_z, selected_z)
            total_consistency += consistency
        
        return total_consistency
            
    def train_adversarial(self, X, X_raw, size_factor, user_selected_lists=None,
                         n_epochs=100, batch_size=256, lr_gen=0.001, lr_disc=0.001,
                         disc_iters=1, gen_iters=1, lambda_cluster=1.0, lambda_recon=1.0,
                         lambda_attention=1.0, lambda_feature=1.0, lambda_consistency=1.0,
                         pretrain_disc_epochs=10, label_smoothing=0.1):
        """
        Train adversarial clustering model
        
        Args:
            X: normalized input matrix [n_samples, n_features]
            X_raw: raw count matrix [n_samples, n_features]
            size_factor: size factors [n_samples,]
            user_selected_lists: list of lists of indices [[group1_indices], [group2_indices], ...]
            n_epochs: number of training epochs
            batch_size: batch size
            lr_gen: learning rate for generator
            lr_disc: learning rate for discriminator
            disc_iters: discriminator iterations per epoch
            gen_iters: generator iterations per epoch
            lambda_cluster: weight for clustering loss
            lambda_recon: weight for reconstruction loss
            lambda_attention: weight for attention loss
            lambda_feature: weight for feature separation loss
            lambda_consistency: weight for consistency loss
            pretrain_disc_epochs: epochs for pretraining discriminator
            label_smoothing: label smoothing factor
            
        Returns:
            y_pred: predicted cluster labels [n_samples,]
            latent: latent representations [n_samples, z_dim]
        """
        # Convert to dense if needed
        X = check_matrix(X)
        X_raw = check_matrix(X_raw)
        
        # Process user_selected_lists
        if user_selected_lists is None:
            user_selected_lists = []
        
        # Convert to list of lists if needed
        if not isinstance(user_selected_lists, list):
            user_selected_lists = [user_selected_lists]
        
        # Ensure each element is a list and filter valid indices
        processed_lists = []
        for user_list in user_selected_lists:
            if user_list is None:
                continue
            if not isinstance(user_list, (list, np.ndarray)):
                user_list = [user_list]
            valid_idx = [i for i in user_list if 0 <= i < X.shape[0]]
            if len(valid_idx) > 0:
                processed_lists.append(valid_idx)
        
        user_selected_lists = processed_lists
        num_groups = len(user_selected_lists)
        
        if num_groups != self.num_user_groups:
            print(f"Warning: Expected {self.num_user_groups} groups, got {num_groups}. "
                  f"Adjusting number of discriminators.")
            self.num_user_groups = num_groups
            # Reinitialize discriminator with correct number of groups
            self.discriminator = MultiDiscriminator(
                latent_dim=self.z_dim,
                n_clusters=self.n_clusters,
                num_groups=num_groups,
                hidden_dims=[128, 64]
            ).to(self.device)
        
        device = self.device
        epsilon = 1e-3
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
        sf_tensor = torch.tensor(size_factor, dtype=torch.float32)

        # Create dataset with indices
        dataset_with_indices = TensorDataset(
            torch.arange(len(X_tensor)),
            X_tensor,
            X_raw_tensor,
            sf_tensor
        )
        dataloader = DataLoader(
            dataset_with_indices,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda"
        )
        
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=20, gamma=0.5)
        disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=20, gamma=0.5)
        
        # Create masks for each user group
        user_masks = []
        user_sets = []  # For efficient lookup
        for user_list in user_selected_lists:
            mask = torch.zeros(X_tensor.shape[0], dtype=torch.bool, device=device)
            if len(user_list) > 0:
                mask[user_list] = True
            user_masks.append(mask)
            user_sets.append(set(user_list))
        
        # Initialize cluster centers
        with torch.no_grad():
            latent_features = self.generator.encodeBatch(X_tensor)
            
            # If there are user-selected groups, reserve centers for them
            if num_groups > 0 and num_groups < self.n_clusters:
                # Calculate centers for user-selected groups
                group_centers = []
                for user_list in user_selected_lists:
                    if len(user_list) > 0:
                        selected_features = latent_features[user_list]
                        center = torch.mean(selected_features, dim=0, keepdim=True)
                        group_centers.append(center.cpu().numpy())
                
                # Create mask for all user-selected samples
                all_selected_mask = torch.zeros(X.shape[0], dtype=bool)
                for user_list in user_selected_lists:
                    all_selected_mask[user_list] = True
                
                # KMeans on remaining samples
                other_features = latent_features[~all_selected_mask].cpu().numpy()
                num_other_clusters = self.n_clusters - len(group_centers)
                
                if other_features.shape[0] >= num_other_clusters > 0:
                    kmeans = KMeans(n_clusters=num_other_clusters, n_init=20, random_state=42)
                    other_centers = kmeans.fit(other_features).cluster_centers_
                    cluster_centers = np.vstack([other_centers] + group_centers)
                elif len(group_centers) > 0:
                    # Not enough non-selected samples, just use group centers
                    cluster_centers = np.vstack(group_centers)
                    # Pad with random centers if needed
                    while cluster_centers.shape[0] < self.n_clusters:
                        random_idx = np.random.choice(latent_features.shape[0])
                        random_center = latent_features[random_idx].cpu().numpy().reshape(1, -1)
                        cluster_centers = np.vstack([cluster_centers, random_center])
                else:
                    # Fallback to standard kmeans
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
                    cluster_centers = kmeans.fit(latent_features.cpu().numpy()).cluster_centers_
                
                self.generator.mu.data.copy_(
                    torch.tensor(cluster_centers[:self.n_clusters], dtype=torch.float32, device=device)
                )
            else:
                # Standard kmeans initialization
                if X.shape[0] >= self.n_clusters:
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
                    centers = kmeans.fit(latent_features.cpu().numpy()).cluster_centers_
                    self.generator.mu.data.copy_(torch.tensor(centers, dtype=torch.float32, device=device))
                else:
                    print(f"Warning: Number of samples ({X.shape[0]}) < n_clusters ({self.n_clusters})")
                    # Use all samples as centers and pad
                    centers = latent_features.cpu().numpy()
                    while centers.shape[0] < self.n_clusters:
                        indices = np.random.choice(centers.shape[0], self.n_clusters - centers.shape[0])
                        padding = centers[indices]
                        centers = np.vstack([centers, padding])
                    self.generator.mu.data.copy_(
                        torch.tensor(centers[:self.n_clusters], dtype=torch.float32, device=device)
                    )
        
        # Pretrain discriminators
        if num_groups > 0 and pretrain_disc_epochs > 0:
            print("Pretraining discriminators...")
            for epoch in range(pretrain_disc_epochs):
                disc_loss_sum = 0
                disc_acc_sum = 0
                total_samples = 0
                
                for batch_idx, (indices, x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(device, non_blocking=True)
                    current_batch_size = x_batch.size(0)
                    total_samples += current_batch_size

                    # Get batch masks for each group
                    batch_masks = []
                    for user_set in user_sets:
                        batch_mask = torch.tensor([idx.item() in user_set for idx in indices], 
                                                 dtype=torch.bool, device=device)
                        batch_masks.append(batch_mask)

                    with torch.no_grad():
                        z, q, _, _, _ = self.generator(x_batch)

                    # Train each discriminator
                    disc_optimizer.zero_grad()
                    attention_preds = self.discriminator(z.detach(), q.detach())
                    
                    disc_loss = torch.tensor(0.0, device=device)
                    correct = 0
                    
                    for group_idx, (attention_pred, batch_mask) in enumerate(zip(attention_preds, batch_masks)):
                        # Generate labels
                        high_attention_labels = torch.ones(current_batch_size, 1, device=device) * (1.0 - label_smoothing)
                        low_attention_labels = torch.ones(current_batch_size, 1, device=device) * label_smoothing
                        target_labels = torch.where(batch_mask.unsqueeze(1), high_attention_labels, low_attention_labels)

                        # Clamp predictions
                        attention_pred_clamped = torch.clamp(attention_pred, min=epsilon, max=1.0 - epsilon)
                        loss = nn.BCELoss()(attention_pred_clamped, target_labels)
                        disc_loss += loss
                        
                        # Calculate accuracy
                        predicted_labels = (attention_pred > 0.5).float()
                        true_labels_binary = batch_mask.float().unsqueeze(1)
                        correct += (predicted_labels == true_labels_binary).sum().item()

                    disc_loss.backward()
                    disc_optimizer.step()

                    disc_loss_sum += disc_loss.item() * current_batch_size
                    disc_acc_sum += correct

                avg_disc_loss = disc_loss_sum / total_samples if total_samples > 0 else 0
                avg_disc_acc = disc_acc_sum / (total_samples * num_groups) if total_samples > 0 else 0
                
                if (epoch + 1) % 10 == 0:
                    print(f"Pretrain Disc Epoch {epoch+1}/{pretrain_disc_epochs}: "
                          f"Loss = {avg_disc_loss:.4f}, Acc = {avg_disc_acc:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Main adversarial training loop
        for epoch in range(n_epochs):
            total_gen_loss = 0
            total_disc_loss = 0
            total_recon_loss = 0
            total_cluster_loss = 0
            total_adv_attn_loss = 0
            total_inner_loss = 0
            total_outer_loss = 0
            total_consistency_loss = 0
            total_samples = 0

            # Train discriminators
            self.discriminator.train()
            self.generator.eval()
            disc_epoch_loss = 0
            disc_epoch_acc = 0
            disc_samples = 0
            
            for _ in range(disc_iters):
                for batch_idx, (indices, x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(device, non_blocking=True)
                    current_batch_size = x_batch.size(0)
                    disc_samples += current_batch_size

                    batch_masks = []
                    for user_set in user_sets:
                        batch_mask = torch.tensor([idx.item() in user_set for idx in indices], 
                                                 dtype=torch.bool, device=device)
                        batch_masks.append(batch_mask)

                    with torch.no_grad():
                        z, q, _, _, _ = self.generator(x_batch)

                    # Train discriminators
                    disc_optimizer.zero_grad()
                    attention_preds = self.discriminator(z.detach(), q.detach())
                    
                    disc_loss = torch.tensor(0.0, device=device)
                    correct = 0
                    
                    for attention_pred, batch_mask in zip(attention_preds, batch_masks):
                        high_attention_labels = torch.ones(current_batch_size, 1, device=device) * (1.0 - label_smoothing)
                        low_attention_labels = torch.ones(current_batch_size, 1, device=device) * label_smoothing
                        target_labels = torch.where(batch_mask.unsqueeze(1), high_attention_labels, low_attention_labels)

                        attention_pred_clamped = torch.clamp(attention_pred, min=epsilon, max=1.0 - epsilon)
                        loss = nn.BCELoss()(attention_pred_clamped, target_labels)
                        disc_loss += loss
                        
                        predicted_labels = (attention_pred > 0.5).float()
                        true_labels_binary = batch_mask.float().unsqueeze(1)
                        correct += (predicted_labels == true_labels_binary).sum().item()

                    disc_loss.backward()
                    disc_optimizer.step()

                    disc_epoch_loss += disc_loss.item() * current_batch_size
                    disc_epoch_acc += correct

            avg_disc_epoch_loss = disc_epoch_loss / disc_samples if disc_samples > 0 else 0
            avg_disc_epoch_acc = disc_epoch_acc / (disc_samples * num_groups) if disc_samples > 0 and num_groups > 0 else 0
            total_disc_loss = avg_disc_epoch_loss

            # Train generator
            self.generator.train()
            self.discriminator.eval()
            gen_epoch_loss = 0
            gen_samples = 0
            
            for _ in range(gen_iters):
                for batch_idx, (indices, x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(device, non_blocking=True)
                    x_raw_batch = x_raw_batch.to(device, non_blocking=True)
                    sf_batch = sf_batch.to(device, non_blocking=True)
                    current_batch_size = x_batch.size(0)
                    gen_samples += current_batch_size
                    total_samples += current_batch_size

                    batch_masks = []
                    for user_set in user_sets:
                        batch_mask = torch.tensor([idx.item() in user_set for idx in indices], 
                                                 dtype=torch.bool, device=device)
                        batch_masks.append(batch_mask)

                    # Generator forward pass
                    z, q, mean_tensor, disp_tensor, pi_tensor = self.generator(x_batch)

                    # 1. Reconstruction loss (ZINB)
                    recon_loss = self.generator.zinb_loss(x=x_raw_batch, mean=mean_tensor,
                                                         disp=disp_tensor, pi=pi_tensor,
                                                         scale_factor=sf_batch)
                    total_recon_loss += recon_loss.item() * current_batch_size

                    # 2. Clustering loss (KL divergence)
                    p = self.generator.target_distribution(q).detach()
                    cluster_loss = self.generator.cluster_loss(p, q)
                    total_cluster_loss += cluster_loss.item() * current_batch_size

                    # 3. Adversarial attention loss (fool discriminators)
                    attention_preds = self.discriminator(z, q)
                    adv_attn_loss = torch.tensor(0.0, device=device)
                    
                    for attention_pred, batch_mask in zip(attention_preds, batch_masks):
                        if torch.any(batch_mask):
                            # Try to make discriminator think selected samples are high attention
                            target_attention_selected = torch.ones_like(attention_pred[batch_mask])
                            attention_pred_clamped = torch.clamp(attention_pred[batch_mask], 
                                                                min=epsilon, max=1.0 - epsilon)
                            adv_attn_loss += nn.BCELoss()(attention_pred_clamped, target_attention_selected)
                    
                    total_adv_attn_loss += adv_attn_loss.item() * current_batch_size if num_groups > 0 else 0

                    # 4. Feature separation loss
                    inner_loss, outer_loss = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                    if lambda_feature > 0 and num_groups > 0:
                        inner_loss, outer_loss = self.feature_separation_loss(z, batch_masks)
                        total_inner_loss += inner_loss.item() * current_batch_size
                        total_outer_loss += outer_loss.item() * current_batch_size

                    # 5. Consistency loss
                    # consistency_loss = torch.tensor(0.0, device=self.device)
                    # if lambda_consistency > 0 and num_groups > 0:
                    #     consistency_loss = self.consistency_loss(z, batch_masks)
                    #     total_consistency_loss += consistency_loss.item() * current_batch_size

                    # Total generator loss
                    gen_loss = (lambda_recon * recon_loss +
                               lambda_cluster * cluster_loss +
                               lambda_attention * adv_attn_loss +
                               lambda_feature * inner_loss +
                               lambda_feature * outer_loss)
                               # lambda_consistency * consistency_loss)

                    # Generator optimization
                    gen_optimizer.zero_grad()
                    gen_loss.backward()
                    gen_optimizer.step()

                    gen_epoch_loss += gen_loss.item() * current_batch_size

            avg_gen_epoch_loss = gen_epoch_loss / gen_samples if gen_samples > 0 else 0
            total_gen_loss = avg_gen_epoch_loss
            
            gen_scheduler.step()
            disc_scheduler.step()

            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Print progress
            if (epoch+1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
                print(f"--- Epoch {epoch+1}/{n_epochs} ---")
                print(f"  Disc Loss: {total_disc_loss:.4f}, Disc Acc: {avg_disc_epoch_acc:.4f}")
                print(f"  Gen Loss: {total_gen_loss:.4f}")
                
                # Print loss components
                avg_recon = (total_recon_loss / total_samples) * lambda_recon if total_samples > 0 else 0
                avg_cluster = (total_cluster_loss / total_samples) * lambda_cluster if total_samples > 0 else 0
                avg_adv_attn = (total_adv_attn_loss / total_samples) * lambda_attention if total_samples > 0 else 0
                avg_inner = (total_inner_loss / total_samples) * lambda_feature if total_samples > 0 else 0
                avg_outer = (total_outer_loss / total_samples) * lambda_feature if total_samples > 0 else 0
                avg_consist = (total_consistency_loss / total_samples) * lambda_consistency if total_samples > 0 else 0
                
                print(f"  Loss Components (Avg Weighted): Recon={avg_recon:.4f}, Cluster={avg_cluster:.4f}, "
                      f"AdvAttn={avg_adv_attn:.4f}")
                print(f"                              Inner={avg_inner:.4f}, Outer={avg_outer:.4f}, "
                      f"Consist={avg_consist:.4f}")
                
                # Show distribution of user-selected samples
                with torch.no_grad():
                    self.generator.eval()
                    self.discriminator.eval()
                    latent_full = self.generator.encodeBatch(X_tensor)
                    q_full = self.generator.soft_assign(latent_full)
                    y_pred = torch.argmax(q_full, dim=1).cpu().numpy()
                    
                    for group_idx, user_list in enumerate(user_selected_lists):
                        if len(user_list) > 0:
                            selected_labels = y_pred[user_list]
                            unique_labels, counts = np.unique(selected_labels, return_counts=True)
                            print(f"  Group {group_idx+1} distribution: {dict(zip(unique_labels, counts))}")
        
        # Get final clustering results
        with torch.no_grad():
            self.generator.eval()
            latent_full = self.generator.encodeBatch(X_tensor)
            q_full = self.generator.soft_assign(latent_full)
            y_pred = torch.argmax(q_full, dim=1).cpu().numpy()
        
        return y_pred, latent_full.cpu().numpy()
