import torch
import pandas as pd
import numpy as np
import os

class DigitalTwinEHRGenerator:
    def __init__(self, num_patients=10000, random_seed=42):
        self.num_patients = num_patients
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    def generate_base_demographics(self):
        # Age distribution: Bimodal (skewed towards older populations for hospital data)
        ages = torch.normal(mean=55.0, std=15.0, size=(self.num_patients,))
        ages = torch.clamp(ages, min=18.0, max=95.0).int().float()
        return ages

    def generate_correlated_vitals(self, ages):
        # BMI is loosely correlated with age, but has its own variance
        bmi_base = torch.normal(mean=26.0, std=5.0, size=(self.num_patients,))
        bmi = bmi_base + (ages * 0.05)
        bmi = torch.clamp(bmi, min=16.0, max=45.0)

        # Blood Pressure (Systolic) strongly correlates with Age and BMI
        sys_bp = 90.0 + (ages * 0.4) + (bmi * 0.6) + torch.normal(mean=0.0, std=10.0, size=(self.num_patients,))
        
        return bmi, sys_bp

    def inject_pandemic_markers(self, ages, bmi):
        # Simulate an outbreak: Probability of infection rises with certain risk factors
        # 0 = Healthy, 1 = Infected (The target variable for our global model)
        risk_score = (ages * 0.02) + (bmi * 0.05) + torch.normal(mean=0.0, std=1.0, size=(self.num_patients,))
        
        # Convert to probabilities using a Sigmoid function, then to binary classes
        infection_prob = torch.sigmoid(risk_score - 3.5) 
        infected_status = torch.bernoulli(infection_prob).int()
        
        return infected_status

    def build_dataset(self, output_path):
        print(f"[*] Initializing Digital Twin EHR Generation for {self.num_patients} nodes...")
        
        ages = self.generate_base_demographics()
        bmi, sys_bp = self.generate_correlated_vitals(ages)
        outbreak_status = self.inject_pandemic_markers(ages, bmi)
        
        # Create unique decentralized IDs for our federated nodes
        patient_ids = [f"AEGIS-NODE-{i:06d}" for i in range(self.num_patients)]
        
        df = pd.DataFrame({
            "Node_ID": patient_ids,
            "Age": ages.numpy().round(1),
            "BMI": bmi.numpy().round(2),
            "Systolic_BP": sys_bp.numpy().round(1),
            "Outbreak_Target": outbreak_status.numpy()
        })
        
        df.to_csv(output_path, index=False)
        print(f"[+] Successfully generated {self.num_patients} correlated patient records.")
        print(f"[+] Data secured at: {output_path}")
        return df

if __name__ == "__main__":
    # Ensure we are saving in the correct directory
    save_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(save_dir, "synthetic_patients_ehr.csv")
    
    generator = DigitalTwinEHRGenerator(num_patients=5000) # Generating 5000 edge nodes
    generator.build_dataset(file_path)