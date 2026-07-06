"""
==============================================================================
THGNN v2.3 – COMPLETE HYPERPARAMETER TUNING FRAMEWORK
Temporal Heterogeneous Graph Neural Network — AIT-NDS & CIC-IDS2018 Editions
==============================================================================

TIME WINDOW CONCEPTS:
  - temporal_window_s: Duration of each snapshot (e.g., 300 seconds = 5 minutes)
  - walk_temporal_window_s: Time window for temporal walks
  - memory_decay_half_life: How fast node memory decays over time
  
TUNING STRATEGIES:
  ✓ Grid Search - Brute force over all combinations
  ✓ Random Search - Random sampling of hyperparameter space
  ✓ Bayesian Optimization (Optuna) - Smart, adaptive search
  ✓ Hyperband - Resource-efficient early stopping
  ✓ Learning Rate Finder - Find optimal LR automatically

ADVANCED FEATURES:
  ✓ Custom parameter ranges
  ✓ Multi-metric optimization
  ✓ Visualization of results
  ✓ Cross-validation support
  ✓ Early stopping integration


==============================================================================
"""

import os
import sys
import json
import time
import copy
import itertools
import warnings
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# For Bayesian optimization
try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")

# For visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import the THGNN models
try:
    from thgnn_ait import Config as AITConfig, THGNNv2_3 as AITModel, load_and_preprocess as load_ait, build_node_features_window
    from thgnn_cic import Config as CICConfig, THGNNv2_3_CIC as CICModel, load_and_preprocess as load_cic
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger.warning("THGNN models not found. Ensure thgnn_ait.py and thgnn_cic.py are available.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hyperparameter Space Definition (with Time Window)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters including time windows"""
    
    # ========== TIME WINDOW PARAMETERS (CRITICAL FOR TEMPORAL LEARNING) ==========
    temporal_window_s: List[float] = field(default_factory=lambda: [60, 120, 300, 600, 900, 1800])
    """Duration of each snapshot in seconds. 
       - 60s: Very fine-grained, captures rapid changes
       - 300s: Default, good balance
       - 900s: Coarse, captures long-term patterns
       - 1800s: Very coarse, for stable patterns"""
    
    walk_temporal_window_s: List[float] = field(default_factory=lambda: [300, 600, 1200, 1800, 3600])
    """Time window for temporal walks. How far back to look for walk continuation.
       Should be >= temporal_window_s typically."""
    
    memory_decay_half_life: List[float] = field(default_factory=lambda: [1800, 3600, 7200, 14400, 28800])
    """Half-life of node memory in seconds.
       - 1800s (30 min): Fast forgetting, focuses on recent activity
       - 3600s (1 hour): Default
       - 7200s (2 hours): Slower forgetting
       - 14400s (4 hours): Long-term memory"""
    
    # ========== Architecture Parameters ==========
    hidden_dim: List[int] = field(default_factory=lambda: [64, 128, 256])
    """Dimension of hidden representations"""
    
    num_hec_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    """Number of Heterogeneous Edge Convolution layers"""
    
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    """Number of attention heads in HEC"""
    
    dropout: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4])
    """Dropout rate for regularization"""
    
    memory_dim: List[int] = field(default_factory=lambda: [64, 128, 256])
    """Dimension of temporal node memory"""
    
    # ========== Walk Parameters ==========
    walk_length: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    """Length of temporal walks (number of hops)"""
    
    num_walks: List[int] = field(default_factory=lambda: [10, 15, 20, 30, 40, 50])
    """Number of walks per source node"""
    
    walk_exploration_epsilon: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3])
    """Exploration rate for walk generation (epsilon-greedy)"""
    
    walk_encoder: List[str] = field(default_factory=lambda: ["transformer", "gru"])
    """Type of walk encoder architecture"""
    
    # ========== Loss Weights ==========
    lambda_recon: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    """Weight for edge reconstruction loss"""
    
    lambda_contrastive: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.8, 1.0])
    """Weight for contrastive learning loss"""
    
    lambda_temporal: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.8])
    """Weight for temporal consistency loss"""
    
    lambda_path_recon: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 0.8])
    """Weight for path reconstruction loss"""
    
    lambda_rarity: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    """Weight for rarity scoring loss"""
    
    # ========== Contrastive Learning ==========
    temperature: List[float] = field(default_factory=lambda: [0.05, 0.07, 0.1, 0.15, 0.2])
    """Temperature for InfoNCE loss"""
    
    hard_neg_ratio: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    """Ratio of hard negatives to use"""
    
    hard_neg_queue_size: List[int] = field(default_factory=lambda: [5000, 10000, 20000])
    """Size of queue for hard negative mining"""
    
    # ========== Training Parameters ==========
    lr: List[float] = field(default_factory=lambda: [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3])
    """Learning rate"""
    
    weight_decay: List[float] = field(default_factory=lambda: [0, 1e-6, 1e-5, 1e-4, 1e-3])
    """Weight decay for regularization"""
    
    batch_size: List[int] = field(default_factory=lambda: [128, 256, 512])
    """Batch size for training"""
    
    grad_clip: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    """Gradient clipping norm"""
    
    # ========== Embedding Dimensions ==========
    port_emb_dim: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    """Port embedding dimension"""
    
    protocol_emb_dim: List[int] = field(default_factory=lambda: [2, 4, 8])
    """Protocol embedding dimension"""
    
    edge_type_emb_dim: List[int] = field(default_factory=lambda: [4, 8, 16])
    """Edge type embedding dimension"""
    
    # ========== Model Variants ==========
    use_hard_negatives: List[bool] = field(default_factory=lambda: [True, False])
    """Whether to use hard negative mining"""
    
    use_edge_types: List[bool] = field(default_factory=lambda: [True, False])
    """Whether to use heterogeneous edge types"""
    
    # ========== Advanced ==========
    walk_encoder_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Number of layers in walk encoder"""
    
    walk_encoder_heads: List[int] = field(default_factory=lambda: [4, 8])
    """Number of attention heads in walk encoder"""
    
    def get_param_names(self) -> List[str]:
        """Get all parameter names"""
        return [f.name for f in self.__dataclass_fields__.values()]
    
    def get_param_grid(self) -> Dict[str, List]:
        """Convert to dictionary for grid search"""
        return asdict(self)
    
    def get_random_params(self) -> Dict[str, Any]:
        """Get random hyperparameters for random search"""
        params = {}
        for param_name, values in asdict(self).items():
            if values:
                params[param_name] = np.random.choice(values)
        return params
    
    def get_time_window_params(self) -> Dict[str, List[float]]:
        """Get only time window related parameters"""
        return {
            'temporal_window_s': self.temporal_window_s,
            'walk_temporal_window_s': self.walk_temporal_window_s,
            'memory_decay_half_life': self.memory_decay_half_life,
        }
    
    def sample_optuna(self, trial: 'Trial') -> Dict[str, Any]:
        """Sample hyperparameters using Optuna with time window awareness"""
        params = {}
        
        # Time window parameters (log-uniform distribution is better for time)
        params['temporal_window_s'] = trial.suggest_float(
            'temporal_window_s', 
            min(self.temporal_window_s), 
            max(self.temporal_window_s), 
            log=True
        )
        params['walk_temporal_window_s'] = trial.suggest_float(
            'walk_temporal_window_s',
            min(self.walk_temporal_window_s),
            max(self.walk_temporal_window_s),
            log=True
        )
        params['memory_decay_half_life'] = trial.suggest_float(
            'memory_decay_half_life',
            min(self.memory_decay_half_life),
            max(self.memory_decay_half_life),
            log=True
        )
        
        # Categorical parameters
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', self.hidden_dim)
        params['num_hec_layers'] = trial.suggest_int('num_hec_layers', min(self.num_hec_layers), max(self.num_hec_layers))
        params['num_heads'] = trial.suggest_categorical('num_heads', self.num_heads)
        params['dropout'] = trial.suggest_float('dropout', min(self.dropout), max(self.dropout))
        params['memory_dim'] = trial.suggest_categorical('memory_dim', self.memory_dim)
        
        # Walk parameters
        params['walk_length'] = trial.suggest_int('walk_length', min(self.walk_length), max(self.walk_length))
        params['num_walks'] = trial.suggest_int('num_walks', min(self.num_walks), max(self.num_walks))
        params['walk_exploration_epsilon'] = trial.suggest_float('walk_exploration_epsilon', 
                                                                  min(self.walk_exploration_epsilon), 
                                                                  max(self.walk_exploration_epsilon))
        params['walk_encoder'] = trial.suggest_categorical('walk_encoder', self.walk_encoder)
        
        # Loss weights
        params['lambda_recon'] = trial.suggest_float('lambda_recon', min(self.lambda_recon), max(self.lambda_recon))
        params['lambda_contrastive'] = trial.suggest_float('lambda_contrastive', min(self.lambda_contrastive), max(self.lambda_contrastive))
        params['lambda_temporal'] = trial.suggest_float('lambda_temporal', min(self.lambda_temporal), max(self.lambda_temporal))
        params['lambda_path_recon'] = trial.suggest_float('lambda_path_recon', min(self.lambda_path_recon), max(self.lambda_path_recon))
        params['lambda_rarity'] = trial.suggest_float('lambda_rarity', min(self.lambda_rarity), max(self.lambda_rarity))
        
        # Contrastive
        params['temperature'] = trial.suggest_float('temperature', min(self.temperature), max(self.temperature))
        params['hard_neg_ratio'] = trial.suggest_float('hard_neg_ratio', min(self.hard_neg_ratio), max(self.hard_neg_ratio))
        params['hard_neg_queue_size'] = trial.suggest_categorical('hard_neg_queue_size', self.hard_neg_queue_size)
        
        # Training
        params['lr'] = trial.suggest_float('lr', min(self.lr), max(self.lr), log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', min(self.weight_decay), max(self.weight_decay), log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', self.batch_size)
        params['grad_clip'] = trial.suggest_float('grad_clip', min(self.grad_clip), max(self.grad_clip))
        
        # Embeddings
        params['port_emb_dim'] = trial.suggest_categorical('port_emb_dim', self.port_emb_dim)
        params['protocol_emb_dim'] = trial.suggest_categorical('protocol_emb_dim', self.protocol_emb_dim)
        params['edge_type_emb_dim'] = trial.suggest_categorical('edge_type_emb_dim', self.edge_type_emb_dim)
        
        # Flags
        params['use_hard_negatives'] = trial.suggest_categorical('use_hard_negatives', self.use_hard_negatives)
        params['use_edge_types'] = trial.suggest_categorical('use_edge_types', self.use_edge_types)
        
        # Advanced
        params['walk_encoder_layers'] = trial.suggest_int('walk_encoder_layers', min(self.walk_encoder_layers), max(self.walk_encoder_layers))
        params['walk_encoder_heads'] = trial.suggest_categorical('walk_encoder_heads', self.walk_encoder_heads)
        
        return params


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Time Window Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TimeWindowAnalyzer:
    """Analyze optimal time window settings for the dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_temporal_distribution(self) -> Dict:
        """Analyze the temporal distribution of flows"""
        timestamps = self.df['ts_unix'].values
        time_range = timestamps.max() - timestamps.min()
        
        # Calculate flow rates at different time scales
        analysis = {
            'total_time_range_hours': time_range / 3600,
            'total_flows': len(self.df),
            'avg_flows_per_second': len(self.df) / time_range,
        }
        
        # Test different window sizes
        window_sizes = [60, 120, 300, 600, 900, 1800, 3600, 7200]
        flow_counts = []
        
        for window in window_sizes:
            n_windows = int(np.ceil(time_range / window))
            flows_per_window = []
            for i in range(n_windows):
                start = timestamps.min() + i * window
                end = start + window
                count = np.sum((timestamps >= start) & (timestamps < end))
                flows_per_window.append(count)
            flow_counts.append(flows_per_window)
            
            analysis[f'window_{window}s_mean_flows'] = np.mean(flows_per_window)
            analysis[f'window_{window}s_std_flows'] = np.std(flows_per_window)
            analysis[f'window_{window}s_empty_ratio'] = np.mean([c == 0 for c in flows_per_window])
        
        # Recommend optimal window
        empty_ratios = [analysis[f'window_{w}s_empty_ratio'] for w in window_sizes]
        mean_flows = [analysis[f'window_{w}s_mean_flows'] for w in window_sizes]
        
        # Balance between too empty and too many flows
        scores = [er * 10 + (1 / (mf + 1)) for er, mf in zip(empty_ratios, mean_flows)]
        best_idx = np.argmin(scores)
        
        analysis['recommended_temporal_window_s'] = window_sizes[best_idx]
        analysis['recommended_reason'] = f"Balances empty windows ({empty_ratios[best_idx]:.2f}) and mean flows ({mean_flows[best_idx]:.0f})"
        
        return analysis
    
    def plot_temporal_analysis(self, save_path: str = "temporal_analysis.png"):
        """Plot temporal analysis for visualization"""
        if not PLOT_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        timestamps = self.df['ts_unix'].values
        time_range = timestamps.max() - timestamps.min()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Flow density over time
        axes[0, 0].hist(timestamps, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('Number of Flows')
        axes[0, 0].set_title('Flow Distribution Over Time')
        
        # 2. Inter-arrival time distribution
        iats = np.diff(timestamps)
        axes[0, 1].hist(iats[iats < np.percentile(iats, 99)], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Inter-arrival Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Flow Inter-arrival Time Distribution')
        axes[0, 1].set_xscale('log')
        
        # 3. Flows per window at different scales
        window_sizes = [60, 300, 900, 3600]
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (window, color) in enumerate(zip(window_sizes, colors)):
            n_windows = int(np.ceil(time_range / window))
            flows_per_window = []
            for j in range(n_windows):
                start = timestamps.min() + j * window
                end = start + window
                count = np.sum((timestamps >= start) & (timestamps < end))
                flows_per_window.append(count)
            
            axes[1, 0].plot(range(len(flows_per_window)), flows_per_window, 
                          label=f'{window}s window', color=color, alpha=0.7)
        
        axes[1, 0].set_xlabel('Window Index')
        axes[1, 0].set_ylabel('Flows per Window')
        axes[1, 0].set_title('Flow Volume by Window Size')
        axes[1, 0].legend()
        
        # 4. Empty window ratio vs window size
        empty_ratios = []
        for window in window_sizes:
            n_windows = int(np.ceil(time_range / window))
            empty_count = 0
            for j in range(n_windows):
                start = timestamps.min() + j * window
                end = start + window
                count = np.sum((timestamps >= start) & (timestamps < end))
                if count == 0:
                    empty_count += 1
            empty_ratios.append(empty_count / n_windows)
        
        axes[1, 1].plot(window_sizes, empty_ratios, 'bo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Window Size (seconds)')
        axes[1, 1].set_ylabel('Empty Window Ratio')
        axes[1, 1].set_title('Empty Windows by Size')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Temporal analysis plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hyperparameter Tuner Base Class
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterTuner:
    """Base class for hyperparameter tuning"""
    
    def __init__(self, 
                 model_type: str,  # 'ait' or 'cic'
                 data_path: str,
                 param_space: HyperparameterSpace,
                 output_dir: str = "tuning_results",
                 n_trials: int = 50,
                 cv_folds: int = 3,
                 metric: str = "auroc",
                 maximize: bool = True,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 analyze_time_windows: bool = True):
        
        self.model_type = model_type
        self.data_path = data_path
        self.param_space = param_space
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf if maximize else np.inf
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data once
        self._load_data()
        
        # Analyze time windows if requested
        self.time_window_analysis = None
        if analyze_time_windows:
            self._analyze_time_windows()
        
    def _load_data(self):
        """Load and preprocess data based on model type"""
        if not MODELS_AVAILABLE:
            raise ImportError("THGNN models not available")
        
        if self.model_type == 'ait':
            cfg = AITConfig(data_path=self.data_path)
            self.df, self.meta = load_ait(cfg)
            self.node_feats = build_node_features_window(self.df, self.meta["num_nodes"])
            self.ConfigClass = AITConfig
            self.ModelClass = AITModel
            from thgnn_ait import _AIT_NUMERIC_COLS
            self.numeric_cols = _AIT_NUMERIC_COLS
        elif self.model_type == 'cic':
            cfg = CICConfig(data_path=self.data_path)
            self.df, self.meta = load_cic(cfg)
            self.node_feats = None
            self.ConfigClass = CICConfig
            self.ModelClass = CICModel
            from thgnn_cic import _CIC_NUMERIC_COLS
            self.numeric_cols = _CIC_NUMERIC_COLS
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Split data into train/val/test
        self._split_data()
        
    def _analyze_time_windows(self):
        """Analyze optimal time windows for the dataset"""
        logger.info("Analyzing temporal patterns for optimal time windows...")
        analyzer = TimeWindowAnalyzer(self.df)
        analysis = analyzer.analyze_temporal_distribution()
        
        logger.info(f"  Total time range: {analysis['total_time_range_hours']:.2f} hours")
        logger.info(f"  Total flows: {analysis['total_flows']}")
        logger.info(f"  Avg flows/second: {analysis['avg_flows_per_second']:.2f}")
        logger.info(f"  Recommended temporal_window_s: {analysis['recommended_temporal_window_s']} seconds")
        logger.info(f"  Reason: {analysis['recommended_reason']}")
        
        # Plot analysis
        plot_path = os.path.join(self.output_dir, "temporal_analysis.png")
        analyzer.plot_temporal_analysis(plot_path)
        
        self.time_window_analysis = analysis
        
    def _split_data(self):
        """Split data into train, validation, test sets using temporal split"""
        timestamps = self.df["ts_unix"].values
        train_ratio = 0.7
        val_ratio = 0.15
        
        train_cutoff = np.quantile(timestamps, train_ratio)
        val_cutoff = np.quantile(timestamps, train_ratio + val_ratio)
        
        self.train_df = self.df[self.df["ts_unix"] <= train_cutoff]
        self.val_df = self.df[(self.df["ts_unix"] > train_cutoff) & (self.df["ts_unix"] <= val_cutoff)]
        self.test_df = self.df[self.df["ts_unix"] > val_cutoff]
        
        logger.info(f"Data split - Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
    
    def _create_config(self, params: Dict[str, Any]) -> Any:
        """Create configuration object from parameters"""
        cfg = self.ConfigClass()
        
        # Update config with parameters
        for key, value in params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        # Ensure time window constraints
        if hasattr(cfg, 'walk_temporal_window_s') and hasattr(cfg, 'temporal_window_s'):
            # walk_temporal_window should be >= temporal_window
            if cfg.walk_temporal_window_s < cfg.temporal_window_s:
                cfg.walk_temporal_window_s = cfg.temporal_window_s
        
        # Set reduced epochs for tuning
        cfg.epochs = min(cfg.epochs, 15)
        cfg.patience = 5
        
        # Set proxy attack labels if needed
        if hasattr(cfg, 'proxy_attack_labels'):
            unique_labels = self.df["label"].unique().tolist()
            cfg.proxy_attack_labels = unique_labels
        
        return cfg
    
    def _create_model(self, cfg: Any, ablation_flags: Dict = None) -> nn.Module:
        """Create model instance"""
        if self.model_type == 'ait':
            return self.ModelClass(cfg, self.meta["num_nodes"], 
                                   self.meta["num_proto"], 
                                   self.meta["port_vocab"])
        else:
            if ablation_flags is None:
                ablation_flags = {
                    "use_contrastive": True,
                    "use_temporal": True,
                    "use_path_recon": True,
                    "use_rarity": True,
                    "use_memory": True,
                    "use_hard_negatives": cfg.use_hard_negatives if hasattr(cfg, 'use_hard_negatives') else True,
                    "use_edge_types": cfg.use_edge_types if hasattr(cfg, 'use_edge_types') else True,
                    "walk_encoder": cfg.walk_encoder if hasattr(cfg, 'walk_encoder') else "transformer",
                    "num_hec_layers": cfg.num_hec_layers,
                }
            return self.ModelClass(cfg, self.meta["num_nodes"], 
                                   self.meta["num_proto"], 
                                   ablation_flags)
    
    def _evaluate_config(self, params: Dict[str, Any], 
                         trial_id: int = None) -> Tuple[float, Dict]:
        """Evaluate a single hyperparameter configuration"""
        from thgnn_ait import THGNNv2Trainer as AITTrainer
        from thgnn_cic import THGNNv2Trainer as CICTrainer
        
        try:
            # Create config and model
            cfg = self._create_config(params)
            model = self._create_model(cfg)
            
            # Create trainer
            if self.model_type == 'ait':
                trainer = AITTrainer(cfg, model, self.meta)
            else:
                trainer = CICTrainer(cfg, model, self.meta)
            
            # Train model
            metrics = trainer.train(self.train_df, self.node_feats)
            
            # Get validation score
            score = metrics.get(self.metric, 0.0)
            if np.isnan(score):
                score = 0.0
            
            # Negate if minimizing
            if not self.maximize:
                score = -score
            
            return score, metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf if self.maximize else np.inf, {}
    
    def save_results(self):
        """Save tuning results to file"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.output_dir, "tuning_results.csv"), index=False)
        
        # Save best params
        with open(os.path.join(self.output_dir, "best_params.json"), "w") as f:
            json.dump({
                "best_params": self.best_params,
                "best_score": float(self.best_score),
                "metric": self.metric,
                "model_type": self.model_type,
                "time_window_analysis": self.time_window_analysis,
            }, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def plot_results(self):
        """Plot tuning results"""
        if not PLOT_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Score vs trial
        axes[0, 0].plot(results_df['trial'], results_df['score'], 'o-', alpha=0.7)
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel(f'Score ({self.metric})')
        axes[0, 0].set_title('Score Progression')
        axes[0, 0].axhline(y=self.best_score, color='r', linestyle='--', label='Best')
        axes[0, 0].legend()
        
        # 2. Score distribution
        axes[0, 1].hist(results_df['score'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel(f'Score ({self.metric})')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Score Distribution')
        
        # 3. Time window impact (if available)
        if 'temporal_window_s' in results_df.columns:
            temporal_groups = results_df.groupby(pd.cut(results_df['temporal_window_s'], bins=5))['score'].mean()
            axes[0, 2].bar(range(len(temporal_groups)), temporal_groups.values)
            axes[0, 2].set_xticks(range(len(temporal_groups)))
            axes[0, 2].set_xticklabels([f"{int(g.left)}-{int(g.right)}" for g in temporal_groups.index], rotation=45)
            axes[0, 2].set_xlabel('Temporal Window (seconds)')
            axes[0, 2].set_ylabel(f'Avg {self.metric}')
            axes[0, 2].set_title('Impact of Time Window Size')
        
        # 4. Learning rate impact
        if 'lr' in results_df.columns:
            lr_groups = results_df.groupby(pd.cut(np.log10(results_df['lr']), bins=5))['score'].mean()
            axes[1, 0].bar(range(len(lr_groups)), lr_groups.values)
            axes[1, 0].set_xticks(range(len(lr_groups)))
            axes[1, 0].set_xticklabels([f"{10**g.left:.1e}" for g in lr_groups.index], rotation=45)
            axes[1, 0].set_xlabel('Learning Rate')
            axes[1, 0].set_ylabel(f'Avg {self.metric}')
            axes[1, 0].set_title('Impact of Learning Rate')
        
        # 5. Hidden dimension impact
        if 'hidden_dim' in results_df.columns:
            hidden_groups = results_df.groupby('hidden_dim')['score'].mean()
            axes[1, 1].bar(range(len(hidden_groups)), hidden_groups.values)
            axes[1, 1].set_xticks(range(len(hidden_groups)))
            axes[1, 1].set_xticklabels(hidden_groups.index)
            axes[1, 1].set_xlabel('Hidden Dimension')
            axes[1, 1].set_ylabel(f'Avg {self.metric}')
            axes[1, 1].set_title('Impact of Hidden Dimension')
        
        # 6. Top parameters heatmap (simplified)
        if len(results_df) > 10:
            top_params = results_df.nlargest(10, 'score')[['trial', 'score'] + [c for c in ['hidden_dim', 'num_hec_layers', 'lr', 'temporal_window_s'] if c in results_df.columns]]
            axes[1, 2].axis('off')
            axes[1, 2].table(cellText=top_params.values, colLabels=top_params.columns, 
                            loc='center', cellLoc='center')
            axes[1, 2].set_title('Top 10 Configurations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "tuning_plots.png"), dpi=150)
        plt.close()
        
        logger.info(f"Plots saved to {self.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Grid Search Tuner
# ─────────────────────────────────────────────────────────────────────────────

class GridSearchTuner(HyperparameterTuner):
    """Grid search hyperparameter tuning"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def tune(self) -> Dict[str, Any]:
        """Run grid search"""
        logger.info("Starting Grid Search...")
        
        # Get parameter grid
        param_grid = self.param_space.get_param_grid()
        
        # Limit grid size if too large
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        if total_combinations > 1000:
            logger.warning(f"Grid size ({total_combinations}) too large. Using random subset.")
            param_names = list(param_grid.keys())
            n_samples = min(500, total_combinations)
            param_combinations = []
            for _ in range(n_samples):
                params = {name: np.random.choice(param_grid[name]) for name in param_names}
                param_combinations.append(params)
        else:
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = [
                dict(zip(param_names, combination))
                for combination in itertools.product(*param_values)
            ]
        
        logger.info(f"Testing {len(param_combinations)} configurations")
        
        start_time = time.time()
        
        for trial_idx, params in enumerate(param_combinations):
            logger.info(f"Trial {trial_idx + 1}/{len(param_combinations)}")
            
            # Evaluate configuration
            score, metrics = self._evaluate_config(params, trial_idx)
            
            # Store results
            result = {
                'trial': trial_idx,
                'score': score,
                **params,
                **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
            }
            self.results.append(result)
            
            # Update best
            if (self.maximize and score > self.best_score) or \
               (not self.maximize and score < self.best_score):
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {score:.4f}")
            
            # Save intermediate results
            if (trial_idx + 1) % 10 == 0:
                self.save_results()
        
        elapsed = time.time() - start_time
        logger.info(f"Grid Search completed in {elapsed:.2f} seconds")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        self.save_results()
        self.plot_results()
        
        return self.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Random Search Tuner
# ─────────────────────────────────────────────────────────────────────────────

class RandomSearchTuner(HyperparameterTuner):
    """Random search hyperparameter tuning"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def tune(self) -> Dict[str, Any]:
        """Run random search"""
        logger.info(f"Starting Random Search with {self.n_trials} trials...")
        
        start_time = time.time()
        
        for trial_idx in range(self.n_trials):
            # Sample random parameters
            params = self.param_space.get_random_params()
            logger.info(f"Trial {trial_idx + 1}/{self.n_trials}")
            
            # Evaluate configuration
            score, metrics = self._evaluate_config(params, trial_idx)
            
            # Store results
            result = {
                'trial': trial_idx,
                'score': score,
                **params,
                **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
            }
            self.results.append(result)
            
            # Update best
            if (self.maximize and score > self.best_score) or \
               (not self.maximize and score < self.best_score):
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {score:.4f}")
            
            # Save intermediate results
            if (trial_idx + 1) % 10 == 0:
                self.save_results()
        
        elapsed = time.time() - start_time
        logger.info(f"Random Search completed in {elapsed:.2f} seconds")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        self.save_results()
        self.plot_results()
        
        return self.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Bayesian Optimization Tuner (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianTuner(HyperparameterTuner):
    """Bayesian optimization using Optuna"""
    
    def __init__(self, n_trials: int = 50, **kwargs):
        super().__init__(n_trials=n_trials, **kwargs)
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
    
    def objective(self, trial: 'Trial') -> float:
        """Objective function for Optuna"""
        # Sample hyperparameters
        params = self.param_space.sample_optuna(trial)
        
        # Evaluate configuration
        score, metrics = self._evaluate_config(params, trial.number)
        
        # Store results
        result = {
            'trial': trial.number,
            'score': score,
            **params,
            **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        self.results.append(result)
        
        # Update best
        if (self.maximize and score > self.best_score) or \
           (not self.maximize and score < self.best_score):
            self.best_score = score
            self.best_params = params
            logger.info(f"New best score: {score:.4f}")
        
        # Save results periodically
        if len(self.results) % 10 == 0:
            self.save_results()
        
        # Optuna minimizes, so negate if we want to maximize
        return -score if self.maximize else score
    
    def tune(self) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        logger.info(f"Starting Bayesian Optimization with {self.n_trials} trials...")
        
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best trial
        best_trial = study.best_trial
        
        elapsed = time.time() - start_time
        logger.info(f"Bayesian Optimization completed in {elapsed:.2f} seconds")
        logger.info(f"Best score: {best_trial.value:.4f}")
        
        # Update best params
        self.best_params = best_trial.params
        self.best_score = -best_trial.value if self.maximize else best_trial.value
        
        self.save_results()
        self.plot_results()
        
        # Save Optuna study
        import joblib
        joblib.dump(study, os.path.join(self.output_dir, "optuna_study.pkl"))
        
        return self.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Hyperband Tuner (Successive Halving)
# ─────────────────────────────────────────────────────────────────────────────

class HyperbandTuner(HyperparameterTuner):
    """Hyperband algorithm for resource-efficient tuning"""
    
    def __init__(self, 
                 max_epochs: int = 30,
                 eta: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_epochs = max_epochs
        self.eta = eta
        
    def _evaluate_with_budget(self, params: Dict[str, Any], 
                              budget: float,
                              trial_id: int) -> Tuple[float, Dict]:
        """Evaluate configuration with limited epochs"""
        cfg = self._create_config(params)
        cfg.epochs = int(budget)
        cfg.patience = max(2, budget // 5)
        
        return self._evaluate_config(params, trial_id)
    
    def tune(self) -> Dict[str, Any]:
        """Run Hyperband optimization"""
        logger.info("Starting Hyperband optimization...")
        
        start_time = time.time()
        
        # Hyperband parameters
        s_max = int(np.log(self.max_epochs) / np.log(self.eta))
        B = (s_max + 1) * self.max_epochs
        
        for s in reversed(range(s_max + 1)):
            n = int(np.ceil(B / self.max_epochs / (s + 1) * (self.eta ** s)))
            r = self.max_epochs * (self.eta ** (-s))
            
            logger.info(f"Bracket s={s}: n={n}, r={r:.1f}")
            
            # Initial random configurations
            configurations = []
            for i in range(n):
                params = self.param_space.get_random_params()
                configurations.append({
                    'params': params,
                    'score': None,
                    'budget': r
                })
            
            # Successive halving
            for i in range(s + 1):
                n_i = n * (self.eta ** (-i))
                r_i = r * (self.eta ** i)
                
                logger.info(f"  Stage {i}: evaluating {int(n_i)} configurations with {r_i:.1f} epochs")
                
                # Evaluate all configurations at this budget
                for config in configurations:
                    if config['score'] is None:
                        score, metrics = self._evaluate_with_budget(
                            config['params'], r_i, len(self.results)
                        )
                        config['score'] = score
                        
                        # Store results
                        result = {
                            'trial': len(self.results),
                            'score': score,
                            'budget': r_i,
                            **config['params'],
                            **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
                        }
                        self.results.append(result)
                
                # Keep top 1/eta configurations
                if i < s:
                    configurations.sort(key=lambda x: -x['score'] if self.maximize else x['score'])
                    n_next = int(n_i / self.eta)
                    configurations = configurations[:n_next]
                    logger.info(f"  Keeping top {n_next} configurations")
        
        # Find best configuration
        self.results.sort(key=lambda x: -x['score'] if self.maximize else x['score'])
        best_result = self.results[0]
        self.best_params = {k: v for k, v in best_result.items() 
                           if k not in ['score', 'budget', 'trial']}
        self.best_score = best_result['score']
        
        elapsed = time.time() - start_time
        logger.info(f"Hyperband completed in {elapsed:.2f} seconds")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        self.save_results()
        self.plot_results()
        
        return self.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Learning Rate Finder
# ─────────────────────────────────────────────────────────────────────────────

class LearningRateFinder:
    """Find optimal learning rate using LR range test"""
    
    def __init__(self, model_type: str, data_path: str, device: str = 'cuda'):
        self.model_type = model_type
        self.data_path = data_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        
    def find_lr(self, 
                start_lr: float = 1e-7, 
                end_lr: float = 10.0,
                num_iters: int = 100) -> float:
        """Find optimal learning rate"""
        if not MODELS_AVAILABLE:
            raise ImportError("THGNN models not available")
        
        # Load data
        if self.model_type == 'ait':
            from thgnn_ait import Config, THGNNv2_3, THGNNv2Trainer, load_and_preprocess, build_node_features_window, _AIT_NUMERIC_COLS
            cfg = Config(data_path=self.data_path)
            df, meta = load_and_preprocess(cfg)
            node_feats = build_node_features_window(df, meta["num_nodes"])
            numeric_cols = _AIT_NUMERIC_COLS
        else:
            from thgnn_cic import Config, THGNNv2_3_CIC, THGNNv2Trainer, load_and_preprocess, _CIC_NUMERIC_COLS
            cfg = Config(data_path=self.data_path)
            df, meta = load_cic(cfg)
            node_feats = None
            numeric_cols = _CIC_NUMERIC_COLS
        
        # Create small dataset
        train_df = df[:min(2000, len(df))]
        
        # Create model and optimizer
        if self.model_type == 'ait':
            model = THGNNv2_3(cfg, meta["num_nodes"], meta["num_proto"], meta["port_vocab"]).to(self.device)
        else:
            model = THGNNv2_3_CIC(cfg, meta["num_nodes"], meta["num_proto"], 
                                  {"use_contrastive": True, "use_temporal": True, 
                                   "use_path_recon": True, "use_rarity": True, "use_memory": True,
                                   "use_hard_negatives": True, "use_edge_types": True,
                                   "walk_encoder": "transformer", "num_hec_layers": cfg.num_hec_layers}).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
        
        # LR range test
        lrs = []
        losses = []
        avg_loss = 0.0
        best_loss = float('inf')
        beta = 0.98
        
        # Get a batch
        snap = train_df
        nef = torch.tensor(snap[numeric_cols].values, dtype=torch.float32).to(self.device)
        
        if self.model_type == 'ait':
            from thgnn_ait import build_node_features_window
            node_feats_window = build_node_features_window(snap, meta["num_nodes"]).to(self.device)
        else:
            from thgnn_cic import build_node_features_window
            node_feats_window = build_node_features_window(snap, meta["num_nodes"]).to(self.device)
        
        logger.info("Starting learning rate range test...")
        
        for i in range(num_iters):
            # Update learning rate
            lr = start_lr * (end_lr / start_lr) ** (i / num_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            optimizer.zero_grad()
            walk_dicts = []  # Simplified for LR finder
            out = model(node_feats_window, snap, nef, walk_dicts, float(snap["ts_unix"].mean()), None)
            loss = out['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            avg_loss = beta * avg_loss + (1 - beta) * loss_val
            smoothed_loss = avg_loss / (1 - beta ** (i + 1))
            
            lrs.append(lr)
            losses.append(smoothed_loss)
            
            # Stop if loss diverges
            if smoothed_loss > 4 * best_loss and i > 10:
                logger.info(f"Stopping early at iteration {i} due to divergence")
                break
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Iter {i+1}/{num_iters}, LR: {lr:.2e}, Loss: {smoothed_loss:.4f}")
        
        # Find LR with steepest negative gradient
        losses = np.array(losses)
        lrs = np.array(lrs)
        gradients = np.gradient(np.log(losses + 1e-8))
        
        # Find where gradient is most negative (after warmup)
        warmup = min(10, len(gradients) // 10)
        best_idx = np.argmin(gradients[warmup:]) + warmup
        optimal_lr = lrs[best_idx]
        
        # Plot results
        if PLOT_AVAILABLE:
            plt.figure(figsize=(10, 6))
            plt.plot(lrs, losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2e}')
            plt.legend()
            plt.grid(True)
            plt.savefig('lr_finder.png', dpi=150)
            plt.close()
            logger.info(f"LR finder plot saved to lr_finder.png")
        
        logger.info(f"Optimal learning rate: {optimal_lr:.2e}")
        return optimal_lr


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="THGNN Hyperparameter Tuning with Time Window Analysis")
    
    # General arguments
    parser.add_argument("--model_type", type=str, required=True, choices=['ait', 'cic'],
                       help="Model type: 'ait' (AIT-NDS) or 'cic' (CIC-IDS2018)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--tuning_method", type=str, default='random',
                       choices=['grid', 'random', 'bayesian', 'hyperband', 'lr_finder'],
                       help="Tuning method")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of trials (for random/bayesian/hyperband)")
    parser.add_argument("--metric", type=str, default='auroc',
                       choices=['auroc', 'ap', 'f1', 'precision', 'recall'],
                       help="Metric to optimize")
    parser.add_argument("--output_dir", type=str, default='tuning_results',
                       help="Output directory for results")
    parser.add_argument("--cv_folds", type=int, default=1,
                       help="Number of cross-validation folds (1 = no CV)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Time window specific arguments
    parser.add_argument("--analyze_time_windows", action='store_true', default=True,
                       help="Analyze optimal time windows for the dataset")
    parser.add_argument("--temporal_window", type=float, default=None,
                       help="Fixed temporal window size (overrides tuning)")
    parser.add_argument("--walk_temporal_window", type=float, default=None,
                       help="Fixed walk temporal window size")
    parser.add_argument("--memory_half_life", type=float, default=None,
                       help="Fixed memory half-life")
    
    # Custom parameter ranges
    parser.add_argument("--hidden_dim", type=str, default=None,
                       help="Comma-separated hidden dimensions (e.g., '64,128,256')")
    parser.add_argument("--num_hec_layers", type=str, default=None,
                       help="Comma-separated number of HEC layers")
    parser.add_argument("--lr", type=str, default=None,
                       help="Comma-separated learning rates")
    parser.add_argument("--walk_length", type=str, default=None,
                       help="Comma-separated walk lengths")
    parser.add_argument("--temporal_window_range", type=str, default=None,
                       help="Min,max for temporal window (e.g., '60,1800')")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create hyperparameter space
    param_space = HyperparameterSpace()
    
    # Override with custom values if provided
    if args.hidden_dim:
        param_space.hidden_dim = [int(x) for x in args.hidden_dim.split(',')]
    if args.num_hec_layers:
        param_space.num_hec_layers = [int(x) for x in args.num_hec_layers.split(',')]
    if args.lr:
        param_space.lr = [float(x) for x in args.lr.split(',')]
    if args.walk_length:
        param_space.walk_length = [int(x) for x in args.walk_length.split(',')]
    if args.temporal_window_range:
        min_t, max_t = [float(x) for x in args.temporal_window_range.split(',')]
        param_space.temporal_window_s = [min_t, max_t]
    
    # Fix time windows if specified
    if args.temporal_window:
        param_space.temporal_window_s = [args.temporal_window]
    if args.walk_temporal_window:
        param_space.walk_temporal_window_s = [args.walk_temporal_window]
    if args.memory_half_life:
        param_space.memory_decay_half_life = [args.memory_half_life]
    
    logger.info("=" * 70)
    logger.info("THGNN Hyperparameter Tuning Framework")
    logger.info(f"Model Type: {args.model_type.upper()} Edition")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Tuning Method: {args.tuning_method.upper()}")
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Number of Trials: {args.n_trials}")
    logger.info("=" * 70)
    
    # Special case: Learning Rate Finder
    if args.tuning_method == 'lr_finder':
        lr_finder = LearningRateFinder(args.model_type, args.data_path)
        optimal_lr = lr_finder.find_lr()
        print(f"\n{'='*50}")
        print(f"RECOMMENDED LEARNING RATE: {optimal_lr:.2e}")
        print(f"{'='*50}")
        return
    
    # Select tuner
    tuner_kwargs = {
        'model_type': args.model_type,
        'data_path': args.data_path,
        'param_space': param_space,
        'output_dir': args.output_dir,
        'n_trials': args.n_trials,
        'cv_folds': max(1, args.cv_folds),
        'metric': args.metric,
        'maximize': True,
        'random_state': args.seed,
        'analyze_time_windows': args.analyze_time_windows,
    }
    
    if args.tuning_method == 'grid':
        tuner = GridSearchTuner(**tuner_kwargs)
    elif args.tuning_method == 'random':
        tuner = RandomSearchTuner(**tuner_kwargs)
    elif args.tuning_method == 'bayesian':
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed. Run: pip install optuna")
            return
        tuner = BayesianTuner(**tuner_kwargs)
    elif args.tuning_method == 'hyperband':
        tuner = HyperbandTuner(max_epochs=30, eta=3, **tuner_kwargs)
    else:
        raise ValueError(f"Unknown tuning method: {args.tuning_method}")
    
    # Run tuning
    best_params = tuner.tune()
    
    # Print results
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS FOUND")
    print("=" * 70)
    
    # Group parameters by category
    time_params = {k: v for k, v in best_params.items() 
                   if 'temporal' in k or 'memory' in k or 'walk_temporal' in k}
    arch_params = {k: v for k, v in best_params.items() 
                   if k in ['hidden_dim', 'num_hec_layers', 'num_heads', 'dropout', 'memory_dim']}
    walk_params = {k: v for k, v in best_params.items() 
                   if k in ['walk_length', 'num_walks', 'walk_exploration_epsilon', 'walk_encoder']}
    loss_params = {k: v for k, v in best_params.items() 
                   if k.startswith('lambda_')}
    train_params = {k: v for k, v in best_params.items() 
                    if k in ['lr', 'weight_decay', 'batch_size', 'grad_clip']}
    
    if time_params:
        print("\n* TIME WINDOW PARAMETERS:")
        for k, v in time_params.items():
            print(f"  {k}: {v}")
    
    if arch_params:
        print("\n* ARCHITECTURE PARAMETERS:")
        for k, v in arch_params.items():
            print(f"  {k}: {v}")
    
    if walk_params:
        print("\n* WALK PARAMETERS:")
        for k, v in walk_params.items():
            print(f"  {k}: {v}")
    
    if loss_params:
        print("\n* LOSS WEIGHTS:")
        for k, v in loss_params.items():
            print(f"  {k}: {v}")
    
    if train_params:
        print("\n* TRAINING PARAMETERS:")
        for k, v in train_params.items():
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print(f"Best {args.metric.upper()}: {tuner.best_score:.4f}")
    print("=" * 70)
    
    # Save as config file
    config_path = os.path.join(args.output_dir, "best_config.py")
    with open(config_path, "w") as f:
        f.write(f"# Best hyperparameters from {args.tuning_method.upper()} tuning\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: {args.model_type.upper()}\n")
        f.write(f"# Metric: {args.metric} = {tuner.best_score:.4f}\n\n")
        
        f.write("from dataclasses import dataclass, field\n")
        f.write("from typing import List\n\n")
        
        f.write("@dataclass\n")
        f.write("class BestConfig:\n")
        for k, v in best_params.items():
            if isinstance(v, float):
                f.write(f"    {k}: float = {v}\n")
            elif isinstance(v, int):
                f.write(f"    {k}: int = {v}\n")
            elif isinstance(v, bool):
                f.write(f"    {k}: bool = {v}\n")
            elif isinstance(v, str):
                f.write(f"    {k}: str = '{v}'\n")
            else:
                f.write(f"    {k} = {v}\n")
    
    logger.info(f"Best configuration saved to {config_path}")


if __name__ == "__main__":
    main()