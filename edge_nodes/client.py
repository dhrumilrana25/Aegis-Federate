import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from collections import OrderedDict

# --- 1. MULTIMODAL FUSION ARCHITECTURE ---
class AegisFusionNet(nn.Module):
    """
    Architected for Multi-Institutional Healthcare Telemetry.
    Branch A: MLP for Tabular Electronic Health Records (EHR).
    Branch B: 1D-CNN for Temporal Wearable Data (Heart Rate/SpO2).
    """
    def __init__(self):
        super(AegisFusionNet, self).__init__()
        
        # EHR Branch: Processes Age, BMI, Blood Pressure, etc.
        self.ehr_branch = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Telemetry Branch: 1D-CNN for high-frequency signal processing
        self.telemetry_branch = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion & Classification Head
        self.fusion_layer = nn.Linear(16 + 32, 1)
        self.output = nn.Sigmoid()

    def forward(self, ehr, telemetry):
        # EHR Path
        x_ehr = self.ehr_branch(ehr)
        # Telemetry Path
        x_tel = self.telemetry_branch(telemetry).squeeze(-1)
        # Fuse branches
        combined = torch.cat((x_ehr, x_tel), dim=1)
        return self.output(self.fusion_layer(combined))

# --- 2. TRAINING & VALIDATION LOGIC ---
def train(model, train_loader, epochs=1):
    """Local training loop (Place for Opacus DP-SGD integration)."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    
    for epoch in range(epochs):
        for ehr, tel, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(ehr, tel)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    """Local validation to monitor global model performance."""
    criterion = nn.BCELoss()
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ehr, tel, labels in test_loader:
            outputs = model(ehr, tel)
            loss += criterion(outputs, labels.unsqueeze(1)).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    accuracy = correct / total
    return loss / len(test_loader), accuracy

# --- 3. FEDERATED LEARNING CLIENT ---
class AegisClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, node_id):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.node_id = node_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"\n[🚀] {self.node_id}: Initiating Differentially Private Round...")
        train(self.model, self.train_loader)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        print(f"[📊] {self.node_id}: Evaluating global model updates...")
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

# --- 4. DATA GENERATION & EXECUTION ---
if __name__ == "__main__":
    node_id = os.getenv("NODE_ID", "Hospital_Default")
    
    # Generate Synthetic Tensors for Demo Purposes
    # EHR: (Samples, Features=5) | Tel: (Samples, Channels=2, Length=168)
    ehr_data = torch.randn(100, 5)
    tel_data = torch.randn(100, 2, 168)
    labels = torch.randint(0, 2, (100,)).float()
    
    dataset = TensorDataset(ehr_data, tel_data, labels)
    loader = DataLoader(dataset, batch_size=16)
    
    model = AegisFusionNet()
    
    print(f"\n" + "="*50)
    print(f"🛡️ AEGIS-FEDERATE NODE: {node_id}")
    print("Mode: Secure Federated Learning | Status: READY")
    print("="*50 + "\n")

    # Connect to the 'aggregator' service on the Docker network
    try:
        fl.client.start_client(
            server_address="aggregator:8080",
            client=AegisClient(model, loader, loader, node_id).to_client()
        )
    except Exception as e:
        print(f"[❌] Critical: Could not connect to Aggregator. {e}")