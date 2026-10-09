import torch
import torch.nn as nn
import pytorch_lightning as pl

from .visual_stream import VisemeEncoder
from .audio_stream import PhonemeEncoder

class JointDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(JointDetector, self).__init__()
        self.learning_rate = learning_rate
        
        # Streams
        self.visual_stream = VisemeEncoder(pretrained=True)
        self.audio_stream = PhonemeEncoder(pretrained=True)
        
        # Fusion Layer (Concatenation + MLP)
        # Visual output: 512, Audio output: 512 -> Concat: 1024
        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, visual_x, audio_x):
        """
        visual_x: [Batch, Time=20, Channels=3, H=112, W=112]
        audio_x:  [Batch, Channels=3, MFCC=40, Time=20]
        """
        vis_emb = self.visual_stream(visual_x)  # [B, 512]
        aud_emb = self.audio_stream(audio_x)    # [B, 512]
        
        # Concatenate along feature dimension
        fused = torch.cat((vis_emb, aud_emb), dim=1)  # [B, 1024]
        
        # MLP
        logits = self.fusion(fused)  # [B, 1]
        return logits

    def training_step(self, batch, batch_idx):
        visual_x, audio_x, labels = batch
        labels = labels.view(-1, 1).float()  # Ensure shape [B, 1]
        
        logits = self(visual_x, audio_x)
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate accuracy for logging
        preds = torch.sigmoid(logits) >= 0.5
        acc = (preds == labels).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
