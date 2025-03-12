# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
from typing import List, Dict, Tuple, Any, Optional, Union
from loguru import logger
import torch.nn.functional as F
import kmeans1d
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from torch.utils.data import DataLoader, Subset


class ParameterUtils:
    """Parameter Processing Utility Class"""
    
    @staticmethod
    def dict_to_vector(net_dict: Dict, include_bn: bool = True) -> torch.Tensor:
        """Convert model parameter dictionary to one-dimensional vector
        
        Args:
            net_dict: Model parameter dictionary
            include_bn: Whether to include batch normalization layer parameters
            
        Returns:
            One-dimensional vector containing all parameters
        """
        vec = []
        for key, param in net_dict.items():
            if not include_bn and (key.split('.')[-1] == 'num_batches_tracked' 
                                  or key.split('.')[-1] == 'running_mean' 
                                  or key.split('.')[-1] == 'running_var'):
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)
    
    @staticmethod
    def vector_to_dict(vec: torch.Tensor, net_dict: Dict) -> Dict:
        """Restore one-dimensional vector to model parameter dictionary
        
        Args:
            vec: Parameter vector
            net_dict: Original parameter dictionary (for shape information)
            
        Returns:
            Updated parameter dictionary
        """
        pointer = 0
        for param in net_dict.values():
            num_param = param.numel()
            param.data = vec[pointer:pointer + num_param].view_as(param).data
            pointer += num_param
        return net_dict
    
    @staticmethod
    def get_update(update: Dict, model: Dict) -> Dict:
        """Calculate parameter update values
        
        Args:
            update: Updated parameters
            model: Original parameters
            
        Returns:
            Parameter difference (update amount)
        """
        return {key: update[key] - model[key] for key in model if 
                not (key.split('.')[-1] in ['num_batches_tracked', 'running_mean', 'running_var'])}


class DistanceMetrics:
    """Distance Calculation Methods"""
    
    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate cosine similarity
        
        Args:
            a, b: Input vectors
            
        Returns:
            Cosine similarity value
        """
        return F.cosine_similarity(a, b, dim=0).item()
    
    @staticmethod
    def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate Euclidean distance
        
        Args:
            a, b: Input vectors
            
        Returns:
            Euclidean distance value
        """
        return torch.dist(a, b, p=2).item()
    
    @staticmethod
    def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Calculate Maximum Mean Discrepancy (MMD)
        
        Args:
            x, y: Input tensors
            sigma: Kernel function parameter
            
        Returns:
            MMD distance
        """
        # Use vectorized operations to optimize computational efficiency
        m, n = x.size(0), y.size(0)
        
        # Calculate kernel matrices
        xx_norm = torch.sum(x * x, dim=1, keepdim=True)
        yy_norm = torch.sum(y * y, dim=1, keepdim=True)
        
        xx_kernel = torch.exp(-((xx_norm + xx_norm.T - 2 * torch.mm(x, x.T)) / (2 * sigma**2)))
        yy_kernel = torch.exp(-((yy_norm + yy_norm.T - 2 * torch.mm(y, y.T)) / (2 * sigma**2)))
        xy_kernel = torch.exp(-((xx_norm + yy_norm.T - 2 * torch.mm(x, y.T)) / (2 * sigma**2)))
        
        # Calculate MMD value
        mmd = (torch.sum(xx_kernel) / (m * (m-1)) + 
               torch.sum(yy_kernel) / (n * (n-1)) - 
               2 * torch.sum(xy_kernel) / (m * n))
        
        return mmd


class ClusteringMethods:
    """Clustering Related Methods"""
    
    def __init__(self):
        self.reset_state()
    
    def reset_state(self):
        """Reset clustering state"""
        self.clusters = []
        self.centroids = []
        self.global_score = []
        self.metric_honests = []  
        self.metric_maliciouses = []
    
    def adaptive_clustering(self, metrics: List[float]) -> Tuple[List[int], float]:
        """Adaptive clustering method
        
        Select appropriate clustering algorithm based on data size
        
        Args:
            metrics: List of metrics to cluster
            
        Returns:
            (Clustering result, Clustering score)
        """
        self.reset_state()
        self.global_score = [0] * len(metrics)
        metrics_array = np.array(metrics)
        
        # Select appropriate clustering method
        if len(metrics) > 5:
            self.clusters, self.centroids = kmeans1d.cluster(metrics, 2)
        else:
            median = np.median(metrics_array)
            self.clusters = [0 if m < median else 1 for m in metrics]
            self.centroids = [np.mean(metrics_array[np.array(self.clusters) == i]) 
                             for i in [0, 1]]
        
        # Determine honest/malicious client categories
        honest = max(set(self.clusters), key=self.clusters.count)
        malicious = 1 - honest  # Binary classification case
        
        # Calculate clustering information
        for i, cluster in enumerate(self.clusters):
            if cluster == honest:
                self.metric_honests.append(metrics[i])
                self.global_score[i] = 1
            else:
                self.metric_maliciouses.append(metrics[i])
        
        # Handle edge cases
        if not self.metric_honests or not self.metric_maliciouses:
            return self.global_score, float('inf')
        
        # Use standard deviation as a more robust radius measure
        honest_radius = np.std(self.metric_honests) + 1e-8
        malicious_radius = np.std(self.metric_maliciouses) + 1e-8
        
        # Alternative calculation using mean absolute deviation
        if honest_radius < 1e-6:  # If std is essentially zero
            honest_radius = np.mean([abs(m - self.centroids[honest]) for m in self.metric_honests]) + 1e-8
        
        if malicious_radius < 1e-6:  # If std is essentially zero
            malicious_radius = np.mean([abs(m - self.centroids[malicious]) for m in self.metric_maliciouses]) + 1e-8
        
        cluster_distance = abs(self.centroids[honest] - self.centroids[malicious]) + 1e-8
        
        # Calculate modified Davies-Bouldin-like index
        quality_score = (honest_radius + malicious_radius) / cluster_distance
        
        # Normalize score to avoid extreme values
        quality_score = min(quality_score, 1000)  # Cap at reasonable maximum
        
        return self.global_score, quality_score


class ModelEvaluator:
    """Model Evaluation Methods"""
    
    def __init__(self, device=None, temperature=1.0):
        """Initialize evaluator
        
        Args:
            device: Computing device
            temperature: Softmax temperature parameter
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp = temperature
    
    def confidence_weight(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate confidence-based weights
        
        Args:
            probs: Probability distribution
            
        Returns:
            Confidence weights
        """
        max_probs, _ = probs.max(dim=1)
        return torch.exp(max_probs - 1)
    
    def compute_kl_divergence(self, 
                             global_model: torch.nn.Module, 
                             local_model: torch.nn.Module, 
                             data_loader: DataLoader) -> float:
        """Calculate KL divergence
        
        Args:
            global_model: Global model
            local_model: Local model
            data_loader: Data loader
            
        Returns:
            Weighted KL divergence
        """
        loss_sum = 0.0
        total_samples = len(data_loader.dataset)
        filtered_samples = 0
        
        # Use no_grad to reduce memory usage
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Optimize computation
                G_probs = F.softmax(global_model(images) / self.temp, dim=1)
                probs = F.log_softmax(local_model(images) / self.temp, dim=1)
                
                mask = (G_probs.argmax(dim=1) == labels)
                mask_sum = mask.sum().item()
                
                if mask_sum > 0:
                    G_probs_filtered = G_probs[mask]
                    probs_filtered = probs[mask]
                    
                    # Vectorized calculation
                    conf_weight = self.confidence_weight(G_probs_filtered)
                    kl_div = F.kl_div(probs_filtered, G_probs_filtered, reduction='none').sum(dim=1)
                    weighted_kl = (kl_div * conf_weight).mean()
                    
                    loss_sum += weighted_kl.item() * mask_sum
                    filtered_samples += mask_sum
        
        if filtered_samples == 0:
            return 0
            
        # Calculate weighting factor
        weight = (total_samples / filtered_samples) ** 0.5
        return loss_sum / filtered_samples * weight


class DefenseMethods:
    """Federated Learning Defense Method Collection"""
    
    def __init__(self, args, local_data, iters, temp=1):
        """Initialize
        
        Args:
            args: Global parameters
            local_data: Local dataset
            iters: Current iteration count
            temp: Softmax temperature parameter
        """
        self.args = args
        self.local_data = local_data
        self.iters = iters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize auxiliary classes
        self.clustering = ClusteringMethods()
        self.evaluator = ModelEvaluator(self.device, temp)
        self.params_util = ParameterUtils()
    
    def create_class_loaders(self, batch_size=None):
        """Create data loaders for each class
        
        Args:
            batch_size: Batch size (optional)
            
        Returns:
            List of data loaders for each class
        """
        batch_size = batch_size or self.args.local_bs
        targets = torch.tensor(self.local_data.targets).clone().detach()
        num_classes = 10  # Assuming MNIST or CIFAR10
        
        return [
            DataLoader(
                Subset(self.local_data, (targets == class_idx).nonzero().squeeze()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ) for class_idx in range(num_classes)
        ]
    
    def save_data(self, data, name_prefix):
        """Save data to file
        
        Args:
            data: Data to save
            name_prefix: File name prefix
        """
        filename = f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_{name_prefix}.npy'
        np.save(filename, np.array(data))
    
    def Adaptive_Agg(self, updates, baseline):
        """Adaptive Aggregation defense method
        
        Two-stage clustering: first based on distance clustering, then based on KL divergence
        
        Args:
            updates: Client update list
            baseline: Global baseline model
            
        Returns:
            Filtered update list
        """
        # Stage 1: Layer-wise distance-based clustering
        cluster_time_start = time.perf_counter()
        
        # Optimize memory usage: reuse model instance
        model = copy.deepcopy(baseline).to(self.device)
        global_vector = self.params_util.dict_to_vector(baseline.state_dict())
        model_dicts = []
        model_vectors = []
        
        # Vectorize model parameters
        for update in updates:
            model.load_state_dict(update)
            model_vector = self.params_util.dict_to_vector(model.state_dict())
            model_vectors.append(model_vector)
            model_dicts.append(copy.deepcopy(model.state_dict()))  # Store only state dict
        
        layer_keys = [k for k in baseline.state_dict().keys() 
                    if not k.endswith('num_batches_tracked') 
                    and not k.endswith('running_mean')
                    and not k.endswith('running_var')]
        
        # Store clustering results and scores for each layer
        layer_clusters = {}
        layer_scores = {}
        
        # Cluster each layer separately
        for layer_key in layer_keys:
            # Extract parameter vector for this layer
            baseline_layer = baseline.state_dict()[layer_key].view(-1)
            model_layer_vectors = []
            
            for model_dict in model_dicts:
                model_layer = model_dict[layer_key].view(-1)
                model_layer_vectors.append(model_layer)
            
            # Calculate distances
            euclidean_distances = []
            cosine_distances = []
            
            for model_layer in model_layer_vectors:
                # Euclidean distance
                euclidean_distances.append(torch.dist(baseline_layer, model_layer, p=2).item())
                
                # Cosine distance
                similarity = F.cosine_similarity(baseline_layer, model_layer, dim=0).item()
                cosine_distances.append(1 - similarity)
            
            # Cluster using each distance metric
            clusters_cos, radius_cos = self.clustering.adaptive_clustering(cosine_distances)
            clusters_euc, radius_euc = self.clustering.adaptive_clustering(euclidean_distances)
            
            # Select better clustering results
            if radius_cos <= radius_euc:
                layer_clusters[layer_key] = clusters_cos
                layer_scores[layer_key] = radius_cos
            else:
                layer_clusters[layer_key] = clusters_euc
                layer_scores[layer_key] = radius_euc
        
        # Select the layer with best clustering quality (lowest score indicates best quality)
        best_layer = min(layer_scores, key=layer_scores.get)
        best_clusters = layer_clusters[best_layer]
        
        # Record information
        logger.info(f"Best clustering layer selected: {best_layer}, clustering score: {layer_scores[best_layer]:.4f}")
        
        # Stage 1: Filter honest clients
        S1_idxs = [idx for idx, val in enumerate(best_clusters) if val == 1]
        S1_updates = [model_dicts[i] for i in S1_idxs]
        
        cluster_time = (time.perf_counter() - cluster_time_start)
        logger.info(f"Layer-wise clustering time: {cluster_time:.4f}s")
        
        # Stage 2: KL divergence based clustering
        kl_time_start = time.perf_counter()
        class_loaders = self.create_class_loaders()
        KL = []
        
        # Calculate KL divergence for each class
        for update_dict in S1_updates:
            model.load_state_dict(update_dict)
            KL_losses = [
                self.evaluator.compute_kl_divergence(baseline, model, class_loader)
                for class_loader in class_loaders
            ]
            KL.append(KL_losses)
        
        # Optimize computation efficiency
        KL_tensor = torch.tensor(KL).to(self.device)
        KL_distances = torch.norm(
            KL_tensor[:, None, :] - KL_tensor[None, :, :], 
            dim=2, p=2
        ).cpu().detach().numpy()
        
        # Perform second stage clustering
        clustering = AgglomerativeClustering(n_clusters=2).fit(KL_distances)
        flag = 1 if np.sum(clustering.labels_) > len(KL_tensor) // 2 else 0
        S2_idxs = [idx for idx, label in enumerate(clustering.labels_) if label == flag]
        
        kl_time = (time.perf_counter() - kl_time_start)
        logger.info(f"KL clustering time: {kl_time:.4f}s")
        
        # Return final filtered results
        S2_updates = [S1_updates[i] for i in S2_idxs]
        logger.info(f"Final client count retained: {len(S2_updates)}/{len(updates)}")
        
        return S2_updates
    
    @staticmethod
    def average_weights(w, marks):
        """Calculate weighted average
        
        Args:
            w: Weight list
            marks: Mark (weight) list
            
        Returns:
            Weighted average result
        """
        w_avg = copy.deepcopy(w[0])
        total_weight = sum(marks)
        
        for key in w_avg.keys():
            w_avg[key] = sum(w[i][key] * marks[i] for i in range(len(w))) / total_weight
            
        return w_avg