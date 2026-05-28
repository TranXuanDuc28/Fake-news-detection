import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    """Transformer classifier using a pre-trained backbone."""
    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout=0.3, num_classes=2, freeze_backbone=False):
        super(TransformerClassifier, self).__init__()
        
        # Load pre-trained transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.transformer.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_size, num_classes)
        )
        
        # Handle backbone freezing
        if freeze_backbone:
            self.freeze_backbone_weights()
            
    def freeze_backbone_weights(self):
        """Freezes all layers in the transformer backbone to speed up training."""
        for param in self.transformer.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone_weights(self):
        """Unfreezes all layers in the transformer backbone for full fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # Pass inputs to transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract representation for the first token ([CLS])
        # shape of sequence_output: (batch_size, seq_len, hidden_size)
        sequence_output = outputs[0]
        cls_representation = sequence_output[:, 0, :]  # shape: (batch_size, hidden_size)
        
        # Pass through the classification head
        logits = self.fc(cls_representation)  # shape: (batch_size, num_classes)
        return logits
