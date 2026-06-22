<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:050816,45:312E81,100:00D9FF&height=245&section=header&text=Aegis-Federate&fontSize=62&fontColor=ffffff&fontAlignY=35&animation=fadeIn&desc=Multimodal%20Privacy-Preserving%20Health%20Intelligence&descSize=19&descAlignY=58" />

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=700&size=23&duration=2700&pause=900&color=00D9FF&center=true&vCenter=true&width=1000&lines=Federated+Learning+for+privacy-sensitive+health+AI.;Move+the+model+to+the+data%2C+not+the+data+to+the+model.;CNN+%2B+MLP+multimodal+risk+prediction.;Flower+%2B+Opacus+%2B+Docker+%2B+Streamlit." alt="Typing SVG" />

<br/>
<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Federated%20Learning-Flower-7C3AED?style=for-the-badge)](https://flower.ai/)
[![Opacus](https://img.shields.io/badge/Differential%20Privacy-Opacus-00D9FF?style=for-the-badge)](https://opacus.ai/)
[![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

<br/>

![GitHub stars](https://img.shields.io/github/stars/dhrumilrana25/Aegis-Federate?style=for-the-badge\&color=00D9FF)
![GitHub forks](https://img.shields.io/github/forks/dhrumilrana25/Aegis-Federate?style=for-the-badge\&color=7C3AED)
![GitHub last commit](https://img.shields.io/github/last-commit/dhrumilrana25/Aegis-Federate?style=for-the-badge\&color=F97316)

<br/>
<br/>

**Architect:** [Dhrumil Rana](https://github.com/dhrumilrana25)
**Affiliation:** University of Texas at Arlington — MS in Data Science
**Focus:** Federated Learning, Differential Privacy, Multimodal Health AI, Distributed Systems

</div>

---

<br/>

<div align="center">

## 🛑 The Problem: The Privacy Paradox

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=3&section=header" width="75%" />

</div>

<br/>

Modern healthcare AI faces a major bottleneck: **data silos**.

Hospitals, clinics, research labs, and wearable health platforms collect valuable data, but that data cannot simply be centralized because it may contain sensitive patient information.

Healthcare AI needs large, diverse datasets to detect global health patterns, but privacy regulations and institutional boundaries restrict raw data sharing.

The result is the **privacy paradox**:

> AI needs more data to become useful, but the most valuable data is often the hardest to move.

<br/>

<table>
<tr>
<td width="50%" valign="top">

### The Traditional Problem

* Raw patient data is centralized
* Privacy and compliance risk increases
* Data sharing agreements become complex
* Institutions lose control over local data
* AI models remain limited by fragmented access

</td>

<td width="50%" valign="top">

### The Aegis-Federate Approach

* Raw data stays inside local institutions
* Models travel to the data
* Only model updates are shared
* Differential Privacy reduces leakage risk
* Global learning happens without centralizing records

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## ✅ The Solution: Aegis-Federate

</div>

<br/>

**Aegis-Federate** is a containerized, privacy-preserving federated learning system for multimodal health intelligence.

Instead of moving sensitive medical data to a central server, Aegis-Federate moves the model to each local hospital node.

Each edge node trains locally on its own data, then sends privacy-aware model updates to a central aggregator. The global model improves across institutions while raw patient data remains local.

<br/>

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=17&duration=2800&pause=900&color=22C55E&center=true&vCenter=true&width=1000&lines=Hospital+Alpha+Data+stays+local.;Hospital+Beta+Data+stays+local.;Only+model+updates+are+aggregated.;Global+health+intelligence+without+centralized+patient+records." alt="Federated Typing" />

</div>

<br/>

---

<br/>

<div align="center">

## 🏗️ Federated System Architecture

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&height=3&section=header" width="75%" />

</div>

<br/>

```mermaid id="p6eod2"
flowchart LR
    A[🏥 Hospital Alpha<br/>Local Health Data] --> B[🧠 Local Model Training]
    C[🏥 Hospital Beta<br/>Local Health Data] --> D[🧠 Local Model Training]

    B --> E[🔒 DP-SGD<br/>Gradient Clipping + Noise]
    D --> F[🔒 DP-SGD<br/>Gradient Clipping + Noise]

    E --> G[🌐 Flower Aggregator<br/>FedAvg Server]
    F --> G

    G --> H[🧬 Global Multimodal Model]
    H --> I[📊 Streamlit Command Center]

    H --> B
    H --> D

    style A fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style C fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style B fill:#1e293b,stroke:#22c55e,color:#ffffff
    style D fill:#1e293b,stroke:#22c55e,color:#ffffff
    style E fill:#312e81,stroke:#a78bfa,color:#ffffff
    style F fill:#312e81,stroke:#a78bfa,color:#ffffff
    style G fill:#1e293b,stroke:#f97316,color:#ffffff
    style H fill:#064e3b,stroke:#22c55e,color:#ffffff
    style I fill:#7f1d1d,stroke:#ef4444,color:#ffffff
```

<br/>

---

<br/>

<div align="center">

## ✨ Key Architectural Highlights

</div>

<br/>

<table>
<tr>
<td width="50%" valign="top">

## 🧠 1. Multimodal Fusion Architecture

Aegis-Federate uses a dual-branch neural network that combines time-series telemetry with structured health records.

### Telemetry Branch

A **1D Convolutional Neural Network** processes high-frequency temporal signals from wearable-style health telemetry.

Example signals:

* Heart rate
* SpO₂
* Respiratory trends
* Time-dependent physiological patterns

### EHR Branch

A **Multi-Layer Perceptron** processes static electronic health record features.

Example features:

* Age
* BMI
* Blood pressure
* Clinical risk indicators

</td>

<td width="50%" valign="top">

## 🔗 Fusion Layer

The model merges both branches through a dense fusion layer.

This allows the model to learn from:

* Temporal health signals
* Static clinical context
* Cross-modal interactions
* Combined patient risk representations

### Why It Matters

Single-modality models can miss important signals.

A multimodal architecture can better capture patient-level risk by learning from both continuous telemetry and structured health features.

</td>
</tr>
</table>

<br/>

```mermaid id="eg2vxk"
flowchart LR
    A[📈 Wearable Telemetry<br/>Heart Rate, SpO2, Signals] --> B[1D-CNN Branch]
    C[📋 EHR Features<br/>Age, BMI, BP] --> D[MLP Branch]

    B --> E[🔗 Fusion Layer]
    D --> E

    E --> F[🧠 Dense Layers]
    F --> G[🎯 Risk Prediction]

    style A fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style C fill:#0f172a,stroke:#38bdf8,color:#ffffff
    style B fill:#1e293b,stroke:#22c55e,color:#ffffff
    style D fill:#1e293b,stroke:#f97316,color:#ffffff
    style E fill:#312e81,stroke:#a78bfa,color:#ffffff
    style F fill:#1e293b,stroke:#00d9ff,color:#ffffff
    style G fill:#064e3b,stroke:#22c55e,color:#ffffff
```

<br/>

---

<br/>

<div align="center">

## 🔒 Differential Privacy with Opacus

</div>

<br/>

Aegis-Federate integrates **Meta Opacus** to reduce the risk of sensitive information leakage from model updates.

The system uses **Differentially Private Stochastic Gradient Descent**, also known as **DP-SGD**.

<br/>

<table>
<tr>
<td width="50%" valign="top">

### What DP-SGD Does

* Clips gradients to limit individual influence
* Adds calibrated noise to updates
* Tracks privacy budget over training
* Reduces membership inference risk
* Reduces model inversion risk

</td>

<td width="50%" valign="top">

### Privacy Parameters

Differential Privacy is controlled through:

* **ε / epsilon:** privacy budget
* **δ / delta:** probability of privacy failure
* **Noise multiplier:** amount of added noise
* **Max gradient norm:** clipping threshold

</td>
</tr>
</table>

<br/>

> Differential Privacy does not mean “zero risk.” It provides a formal mathematical privacy guarantee that bounds the influence of any single training example on the final model.

<br/>

---

<br/>

<div align="center">

## 🏥 Containerized Cross-Silo Orchestration

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=17&duration=2600&pause=800&color=F97316&center=true&vCenter=true&width=900&lines=Dockerized+Aggregator.;Isolated+Hospital+Nodes.;Federated+Rounds.;Privacy-Utility+Dashboard." alt="Docker Typing" />

</div>

<br/>

Aegis-Federate is structured as a distributed system using Docker Compose.

<table>
<tr>
<td width="33%" valign="top">

### 🌐 Aggregator

The central coordination server.

**Responsibilities**

* Orchestrates federated rounds
* Aggregates client updates
* Runs FedAvg
* Distributes global model weights

</td>

<td width="33%" valign="top">

### 🏥 Edge Nodes

Independent hospital clients.

**Examples**

* Hospital Alpha
* Hospital Beta

**Responsibilities**

* Keep data local
* Train local models
* Apply DP-SGD
* Send model updates only

</td>

<td width="33%" valign="top">

### 📊 Command Center

Real-time monitoring dashboard.

**Responsibilities**

* Visualize training rounds
* Track privacy-utility tradeoff
* Monitor convergence
* Display risk clusters

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## 📊 Interactive Command Center

</div>

<br/>

The Streamlit dashboard acts as a **Privacy-Utility Simulator** for federated health AI.

Users can:

* Adjust the **privacy budget ε**
* Observe the tradeoff between privacy and model performance
* Monitor federated training rounds
* Track global convergence
* Visualize differentially private risk clusters
* Explore interactive Plotly charts and maps

<br/>

```mermaid id="zn75ig"
sequenceDiagram
    participant User
    participant Dashboard as Streamlit Command Center
    participant Server as Flower Aggregator
    participant Alpha as Hospital Alpha
    participant Beta as Hospital Beta

    User->>Dashboard: Adjust privacy budget ε
    Dashboard->>Server: Update simulation settings
    Server->>Alpha: Send global model
    Server->>Beta: Send global model
    Alpha->>Alpha: Local training + DP-SGD
    Beta->>Beta: Local training + DP-SGD
    Alpha->>Server: Send private model update
    Beta->>Server: Send private model update
    Server->>Server: FedAvg aggregation
    Server->>Dashboard: Return convergence metrics
    Dashboard->>User: Visualize privacy-utility tradeoff
```

<br/>

---

<br/>

<div align="center">

## 🛠️ Tech Stack

<br/>

<img src="https://skillicons.dev/icons?i=python,pytorch,docker,streamlit,github,git" />

<br/>
<br/>

</div>

<table>
<tr>
<td width="33%" valign="top">

### 🤖 Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![CNN](https://img.shields.io/badge/1D--CNN-Telemetry%20Branch-7C3AED?style=for-the-badge)
![MLP](https://img.shields.io/badge/MLP-EHR%20Branch-2563EB?style=for-the-badge)

</td>

<td width="33%" valign="top">

### 🔐 Privacy + FL

![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-00D9FF?style=for-the-badge)
![Opacus](https://img.shields.io/badge/Opacus-DP--SGD-22C55E?style=for-the-badge)
![FedAvg](https://img.shields.io/badge/FedAvg-Aggregation-F97316?style=for-the-badge)

</td>

<td width="33%" valign="top">

### 🏗️ Infrastructure

![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white)

</td>
</tr>
</table>

<br/>

---

<br/>

<div align="center">

## 🚀 Quick Start

</div>

<br/>

### 1. Clone the Repository

```bash id="ddp0ve"
git clone https://github.com/dhrumilrana25/Aegis-Federate.git
cd Aegis-Federate
```

### 2. Build and Start the Federated System

```bash id="1bqc5m"
docker compose up --build
```

### 3. Start Federated Training

Depending on your project entrypoint, run the aggregator and clients through Docker Compose or separate terminals.

Example local flow:

```bash id="oovl2y"
python server.py
```

```bash id="zgwmcc"
python client_alpha.py
```

```bash id="r4y8z7"
python client_beta.py
```

### 4. Launch the Dashboard

```bash id="xttcmn"
streamlit run dashboard.py
```

<br/>

---

<br/>

<div align="center">

## 📁 Suggested Repository Structure

</div>

<br/>

```txt id="hzug3t"
Aegis-Federate/
│
├── server/
│   ├── server.py
│   └── aggregation.py
│
├── clients/
│   ├── hospital_alpha.py
│   ├── hospital_beta.py
│   └── client_utils.py
│
├── models/
│   ├── multimodal_model.py
│   ├── telemetry_cnn.py
│   └── ehr_mlp.py
│
├── privacy/
│   ├── dp_engine.py
│   └── privacy_accounting.py
│
├── dashboard/
│   ├── app.py
│   └── visualizations.py
│
├── data/
│   ├── hospital_alpha/
│   └── hospital_beta/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

<br/>

---

<br/>

<div align="center">

## 🔐 Threat Model

</div>

<br/>

| Risk                         | Mitigation                                             |
| ---------------------------- | ------------------------------------------------------ |
| Raw patient data exposure    | Data remains local to each hospital node               |
| Centralized data breach      | No central raw-data lake is created                    |
| Model inversion attacks      | DP-SGD reduces leakage from gradients                  |
| Membership inference attacks | Privacy accounting bounds individual contribution risk |
| Cross-client data leakage    | Clients train in isolated containers                   |
| Uncontrolled collaboration   | Federated server coordinates only model updates        |

<br/>

---

<br/>

<div align="center">

## 📈 Metrics Tracked

</div>

<br/>

Aegis-Federate can be monitored through the Command Center dashboard.

| Metric                | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| **Federated Round**   | Tracks global training progress                       |
| **Local Loss**        | Measures each hospital node’s training behavior       |
| **Global Accuracy**   | Evaluates aggregated model performance                |
| **Privacy Budget ε**  | Tracks privacy-utility tradeoff                       |
| **Noise Multiplier**  | Controls differential privacy strength                |
| **Convergence Curve** | Visualizes model improvement over rounds              |
| **Risk Cluster Map**  | Shows differentially private risk group visualization |

<br/>

---

<br/>

<div align="center">

## 🧭 Roadmap

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=17&duration=2600&pause=800&color=00D9FF&center=true&vCenter=true&width=850&lines=Cross-silo+FL+today.;Secure+aggregation+next.;Kubernetes-scale+federation+future." alt="Roadmap Typing" />

</div>

<br/>

### Current

* [x] Cross-silo federated learning
* [x] Two-node Docker orchestration
* [x] Multimodal CNN + MLP model
* [x] Differential Privacy with Opacus
* [x] Streamlit command center
* [x] Plotly-based interactive visualizations

### Next

* [ ] Secure Multi-Party Computation
* [ ] Secure aggregation protocol
* [ ] More client nodes
* [ ] Model checkpoint registry
* [ ] Advanced privacy accounting
* [ ] Client failure simulation

### Scale

* [ ] Kubernetes orchestration
* [ ] Horizontal scaling to thousands of edge nodes
* [ ] Federated monitoring service
* [ ] Production-style API gateway
* [ ] Deployment-ready FL infrastructure

<br/>

---

<br/>

<div align="center">

## ⚖️ Responsible Use & Compliance Note

</div>

<br/>

This project is designed for educational and research purposes in privacy-preserving machine learning.

Aegis-Federate demonstrates techniques relevant to privacy-sensitive health AI, including federated learning and differential privacy. However, this repository should not be treated as a certified HIPAA, GDPR, or clinical compliance system without formal legal, security, and institutional review.

Do not use real patient data unless you have proper authorization, governance, security controls, and compliance approval.

<br/>

---

<br/>

<div align="center">

## 📈 Repository Stats

<br/>

<img src="https://github-readme-stats.vercel.app/api/pin/?username=dhrumilrana25&repo=Aegis-Federate&theme=tokyonight&hide_border=true" />

<br/>
<br/>

<img src="https://github-readme-activity-graph.vercel.app/graph?username=dhrumilrana25&theme=tokyo-night&hide_border=true&area=true" />

</div>

<br/>

---

<br/>

<div align="center">

## 👨‍💻 Developer

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=700&size=20&duration=2600&pause=900&color=00D9FF&center=true&vCenter=true&width=850&lines=Dhrumil+Rana.;AI+Systems+%26+Machine+Learning+Engineer.;MS+Data+Science+%40+UT+Arlington.;Building+privacy-preserving+AI+systems." alt="Developer Typing" />

<br/>
<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Dhrumil%20Rana-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/dhrumil-rana-265316232/)
[![Email](https://img.shields.io/badge/Email-dhrumilrana25%40gmail.com-D14836?style=for-the-badge\&logo=gmail\&logoColor=white)](mailto:dhrumilrana25@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-dhrumilrana25-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/dhrumilrana25)

<br/>
<br/>

### “Move intelligence across institutions without moving sensitive data.”

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00D9FF,50:312E81,100:050816&height=140&section=footer&animation=twinkling" />

</div>
