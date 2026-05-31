import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for text classification."""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.3, num_classes=2, bidirectional=True):
        super(BiLSTMClassifier, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM (Bidirectional or Unidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected output head
        # The hidden state size is hidden_dim * 2 if bidirectional, else hidden_dim
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)  # Apply dropout on embeddings to prevent memorizing specific words
        
        # LSTM output
        # out shape: (batch_size, seq_len, hidden_dim * 2)
        out, (hn, cn) = self.lstm(embedded)
        
        # Perform global max pooling over the sequence dimension to capture the most salient features
        # out shape after transpose: (batch_size, hidden_dim * 2, seq_len)
        pooled, _ = torch.max(out, dim=1)  # shape: (batch_size, hidden_dim * 2)
        
        # Pass to classification head
        logits = self.fc(pooled)  # shape: (batch_size, num_classes)
        return logits
