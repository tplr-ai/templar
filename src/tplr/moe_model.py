import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, base_config, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        # Shared Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_config["d_model"], nhead=base_config["nhead"]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=base_config["num_layers"])
        
        # Gating network: produces scores for each expert.
        self.gate = nn.Linear(base_config["d_model"], num_experts)
        
        # Define experts: here each expert is a simple linear layer.
        self.experts = nn.ModuleList([
            nn.Linear(base_config["d_model"], base_config["d_model"]) 
            for _ in range(num_experts)
        ])
        
        # Final output head.
        self.head = nn.Linear(base_config["d_model"], base_config["vocab_size"])

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        encoded = self.encoder(x)  # (seq_len, batch_size, d_model)
        # Use mean pooling over sequence as the representation.
        rep = encoded.mean(dim=0)  # (batch_size, d_model)
        
        # Compute gating logits and softmax to get expert weights.
        gating_logits = self.gate(rep)  # (batch_size, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)
        # Hard routing: choose the expert with highest weight.
        expert_indices = torch.argmax(gating_weights, dim=-1)  # (batch_size,)
        
        # Dispatch each sample to its assigned expert.
        expert_outputs = torch.zeros_like(rep)
        for i in range(self.num_experts):
            mask = (expert_indices == i).float().unsqueeze(1)  # (batch_size, 1)
            if mask.sum() > 0:
                expert_out = self.experts[i](rep)
                expert_outputs += mask * expert_out
        logits = self.head(expert_outputs)  # (batch_size, vocab_size)
        return logits, gating_weights, expert_indices
