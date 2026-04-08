# 🛡️ Aegis-Federate: Multimodal Privacy-Preserving Health Intelligence

**Architect:** [Dhrumil Rana](https://github.com/dhrumilrana25)  
**Affiliation:** University of Texas at Arlington, MS in Data Science  
**Tech Stack:** PyTorch, Flower (flwr), Meta Opacus, Docker, Streamlit, Plotly

---

## 🛑 The Problem: The Privacy Paradox & Data Silos
In modern healthcare, AI training faces a critical bottleneck: **Data Silos**. 
To detect global pathogens, AI requires massive datasets. However, strict privacy regulations like **HIPAA** and **GDPR** make it illegal to centralize sensitive patient data. 

**The result?** AI models remain "blind" to global patterns because the data is locked inside individual hospital firewalls.

## ✅ The Solution: Aegis-Federate
Aegis-Federate is a production-grade, containerized infrastructure that solves the privacy paradox. Instead of moving sensitive data to a central server, we **move the AI model to the data**. 

By utilizing **Federated Learning** and **Differential Privacy**, Aegis-Federate enables multi-institutional collaboration without a single byte of raw patient telemetry ever leaving the local hospital's secure storage.

---

## ✨ Key Architectural Highlights (The "Outlier" Features)

### 🧠 1. Multimodal Fusion Architecture (CNN + MLP)
Unlike standard models, Aegis-Federate employs a dual-branch neural network:
* **The Telemetry Branch:** A 1D-Convolutional Neural Network (CNN) designed to process high-frequency temporal data from wearables (Heart Rate, SpO2).
* **The EHR Branch:** A Multi-Layer Perceptron (MLP) for static Electronic Health Records (Age, BMI, Blood Pressure).
* **The Fusion Layer:** A high-level dense layer that merges these modalities for superior pathogen risk prediction.

### 🔒 2. Differential Privacy (Meta Opacus)
To prevent "Model Inversion" attacks, we integrated the **Opacus** engine. By implementing **DP-SGD**, we inject controlled mathematical noise into the gradients. This ensures $(\epsilon, \delta)$-Differential Privacy, making it cryptographically impossible to trace a global model update back to an individual patient.

### 🏗️ 3. Containerized Orchestration (Docker Microservices)
The project is architected as a distributed system using Docker Compose:
* **Aggregator:** The central "brain" orchestrating the secure FedAvg protocol.
* **Edge Nodes (Hospital Alpha & Beta):** Isolated containers representing independent medical centers.
* **Command Center:** A real-time dashboard for monitoring global health metrics.

---

## 📊 Interactive Command Center
The dashboard is a live **Privacy-Utility Simulator**. Users can:
* Adjust the **Privacy Budget ($\epsilon$)** to see the trade-off between cryptographic security and model accuracy.
* Monitor **Federated Rounds** and global model convergence in real-time.
* Visualize **Differentially Private Risk Clusters** via interactive Plotly maps.

---

🎯 Project Goals & Future Roadmap

Current: Cross-Silo FL with 2-node Docker orchestration.

Next: Integration of Secure Multi-Party Computation (SMPC).

Scale: Horizontal scaling to 5,000+ edge devices using Kubernetes.
