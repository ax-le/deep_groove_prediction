import torch
import torch.nn as nn

class MusicAestheticsModel(nn.Module):
    def __init__(self, input_dim=23040, bottleneck_dim=256, hidden_dim=64):
        super().__init__()
        self.metrics = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]
        
        # Shared Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, bottleneck_dim),
            nn.ReLU(),
            nn.LayerNorm(bottleneck_dim)
        )
        
        # Multi-head Experts
        self.heads = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for n in self.metrics
        })

    def forward(self, x):
        feat = self.bottleneck(x)
        return {n: self.heads[n](feat) for n in self.heads}
