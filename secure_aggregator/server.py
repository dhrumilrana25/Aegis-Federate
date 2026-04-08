import flwr as fl
import sys

# --- FEDERATED STRATEGY CONFIGURATION ---
# We use Federated Averaging (FedAvg) to combine weights without seeing raw data.
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,             # Use 100% of available clients for training
    fraction_evaluate=1.0,        # Use 100% of available clients for evaluation
    min_fit_clients=2,            # Wait for at least 2 hospitals to connect
    min_available_clients=2,      # Minimum clients required to start the round
)

def start_aegis_server():
    print("\n" + "="*60)
    print("🛡️  AEGIS-FEDERATE SECURE AGGREGATOR : ONLINE")
    print("Drive Location: D:\\WSL (High-Performance Storage)")
    print("Network: 0.0.0.0:8080 | Mode: Differential Privacy Enabled")
    print("="*60 + "\n")
    
    # Start the gRPC server
    # Address 0.0.0.0 is required for Docker Service Discovery
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    try:
        start_aegis_server()
    except KeyboardInterrupt:
        print("\n[!] Aggregator shutting down gracefully...")
        sys.exit(0)