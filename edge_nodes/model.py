import torch
import torch.nn as nn
import torch.nn.functional as F

class AegisMultimodalPredictor(nn.Module):
    def __init__(self, tabular_input_dim=3, telemetry_channels=2, seq_length=168):
        super(AegisMultimodalPredictor, self).__init__()
        
        # ---------------------------------------------------
        # BRANCH 1: Tabular Encoder (EHR Data)
        # Inputs: Age, BMI, Systolic BP
        # ---------------------------------------------------
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # ---------------------------------------------------
        # BRANCH 2: Temporal Encoder (Wearable Telemetry)
        # Inputs: [Batch, Channels (HR, SpO2), Sequence (168 hrs)]
        # ---------------------------------------------------
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=telemetry_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces seq length to 84
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces seq length to 42
            
            # Global Average Pooling flattens the temporal dimension cleanly
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # ---------------------------------------------------
        # THE FUSION LAYER: Combining Patient Profile + Vitals
        # ---------------------------------------------------
        # Tabular output (16) + Temporal output (32) = 48 concatenated features
        self.fusion_head = nn.Sequential(
            nn.Linear(16 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Single output logit for Outbreak Probability
        )

    def forward(self, tabular_x, telemetry_x):
        # 1. Encode Tabular Data
        tab_features = self.tabular_encoder(tabular_x)
        
        # 2. Encode Temporal Data (Ensure it's shaped [B, C, L])
        temp_features = self.temporal_encoder(telemetry_x)
        temp_features = temp_features.view(temp_features.size(0), -1) # Flatten the pooling output
        
        # 3. Fuse and Predict
        fused = torch.cat((tab_features, temp_features), dim=1)
        out = self.fusion_head(fused)
        
        # Note: We return logits. We will use BCEWithLogitsLoss during training 
        # for numerical stability rather than applying Sigmoid here.
        return out

if __name__ == "__main__":
    # --- Local Sanity Check ---
    # Simulating a batch of 4 patients
    dummy_tabular = torch.randn(4, 3) 
    dummy_telemetry = torch.randn(4, 2, 168) # 2 channels (HR, SpO2), 168 hours
    
    model = AegisMultimodalPredictor()
    output = model(dummy_tabular, dummy_telemetry)
    
    print("[*] Aegis Architecture successfully initialized.")
    print(f"[*] Output shape (Logits): {output.shape}") 
    # Expected output: torch.Size([4, 1])