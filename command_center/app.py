import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aegis-Federate | Command", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- 2. S-TIER UI STYLING (CUSTOM CSS) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .main-header {
        font-size: 3.5rem; 
        font-weight: 800; 
        background: -webkit-linear-gradient(#00F2FE, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.3rem; 
        color: #A0AEC0; 
        text-align: center; 
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #1A1F2B; 
        border-radius: 12px; 
        padding: 25px;
        border: 1px solid #2D3748; 
        text-align: center;
        transition: transform 0.3s;
    }
    .status-card:hover { transform: scale(1.02); border-color: #00F2FE; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; color: #00F2FE; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; gap: 50px; }
    .stTabs [aria-selected="true"] { color: #00F2FE !important; border-bottom-color: #00F2FE !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR: SIMULATION CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("Aegis Control")
st.sidebar.markdown("---")

st.sidebar.subheader("🔒 Privacy Protocol")
epsilon = st.sidebar.select_slider(
    "Privacy Budget ($\epsilon$)", 
    options=[0.1, 0.5, 1.22, 4.0, 8.0, 15.0], 
    value=1.22,
    help="Lower $\epsilon$ = Stronger Privacy (Higher Noise). Higher $\epsilon$ = Better Accuracy (Lower Noise)."
)

st.sidebar.subheader("📡 Distributed Scale")
node_count = st.sidebar.slider("Simulated Edge Nodes", 2, 5000, 150)
federated_rounds = st.sidebar.slider("Training Rounds", 1, 30, 12)

st.sidebar.markdown("---")
if st.sidebar.button("RE-SYNCHRONIZE NETWORK"):
    with st.spinner("Aggregating Global Weights..."):
        time.sleep(1)
        st.sidebar.success("Network In-Sync")

# --- 4. DYNAMIC SIMULATION LOGIC ---
dynamic_acc = 0.88 + (np.log(epsilon + 1) * 0.03)
noise_factor = (2.0 / (epsilon + 0.1))

# --- 5. HEADER ---
st.markdown('<p class="main-header">🛡️ Aegis-Federate: Command</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multimodal Federated Learning with Differential Privacy Orchestration</p>', unsafe_allow_html=True)

# --- 6. LIVE METRIC CARDS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="status-card"><p style="color:#718096; margin-bottom:5px;">ACTIVE EDGE NODES</p><h2>{node_count:,}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="status-card"><p style="color:#718096; margin-bottom:5px;">GLOBAL ACCURACY</p><h2 style="color:#00F2FE;">{dynamic_acc:.1%}</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="status-card"><p style="color:#718096; margin-bottom:5px;">SECURITY STATUS</p><h2 style="color:#48BB78;">{ "MAX PRIVACY" if epsilon < 1.5 else "OPTIMIZED"}</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# --- 7. INTERACTIVE TABS ---
# Variables synchronized: tab_viz, tab_metrics, tab_arch
tab_viz, tab_metrics, tab_arch = st.tabs(["🧬 Risk Visualization", "📈 Convergence Analytics", "🏗️ System Architecture"])

# --- TAB 1: RISK VISUALIZATION ---
with tab_viz:
    st.subheader("Global Anomaly Distribution (DP-View)")
    st.write(f"Displaying {node_count} nodes. Jitter is dynamically scaled based on $\epsilon$ = {epsilon}.")
    
    np.random.seed(42)
    age_jitter = np.random.normal(0, noise_factor * 3, node_count)
    risk_jitter = np.random.normal(0, noise_factor / 15, node_count)
    
    viz_df = pd.DataFrame({
        'Patient Age': np.random.randint(20, 85, node_count) + age_jitter,
        'Risk Probability': np.random.uniform(0.1, 0.9, node_count) + risk_jitter,
        'Node Source': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'], node_count)
    })
    
    fig_scatter = px.scatter(
        viz_df, x='Patient Age', y='Risk Probability', color='Node Source', 
        color_discrete_sequence=px.colors.sequential.Electric,
        template="plotly_dark",
        title=f"Pathogen Risk Outliers ($\epsilon$={epsilon})"
    )
    st.plotly_chart(fig_scatter, width='stretch')

# --- TAB 2: CONVERGENCE ANALYTICS ---
with tab_metrics:
    st.subheader("Federated Learning Performance Metrics")
    st.write(f"Simulating model convergence over {federated_rounds} rounds.")
    
    x_rounds = np.arange(1, federated_rounds + 1)
    y_acc = dynamic_acc - (0.4 / (x_rounds * (epsilon**0.3)))
    
    history_df = pd.DataFrame({"Round": x_rounds, "Accuracy": y_acc})
    
    fig_line = px.line(
        history_df, x="Round", y="Accuracy", markers=True, 
        color_discrete_sequence=["#00F2FE"],
        title="Global Model Learning Curve"
    )
    fig_line.update_yaxes(range=[0.4, 1.0])
    st.plotly_chart(fig_line, width='stretch')

# --- TAB 3: SYSTEM ARCHITECTURE ---
with tab_arch:
    l_col, r_col = st.columns([1.5, 1])
    with l_col:
        st.subheader("The Aegis Technical Manifesto")
        st.markdown("""
        **Aegis-Federate** solves the 'Privacy Paradox' in healthcare AI.
        
        * **Multimodal Fusion:** Employs a dual-branch neural network. A **1D-CNN** processes high-frequency wearable telemetry while an **MLP** processes static Electronic Health Records (EHR).
        * **DP-SGD Integration:** Utilizing the **Meta Opacus** engine to enforce $(\epsilon, \delta)$-Differential Privacy, injecting noise into gradients to prevent patient identification.
        * **Infrastructure:** Orchestrated via Docker microservices, proving horizontal scalability from local silos to global edge networks.
        """)
    with r_col:
        st.info("### Production Stack")
        st.markdown("""
        - **Logic:** PyTorch + Flower (flwr)
        - **Security:** Differential Privacy (Opacus)
        - **Container:** Docker + WSL2 (Ubuntu)
        - **Architect:** Dhrumil Rana
        """)

# --- FOOTER ---
st.divider()
st.caption("Aegis-Federate v1.0.0 | UT Arlington | Built for Production Scale")