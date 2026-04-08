import pandas as pd
import numpy as np
import os
import time

class WearableTelemetryGenerator:
    def __init__(self, ehr_path, hours=168, random_seed=42):
        self.ehr_path = ehr_path
        self.hours = hours # 168 hours = 7 days
        self.np_rng = np.random.default_rng(random_seed)
        
    def generate_telemetry(self, output_path):
        print("[*] Loading Digital Twin EHR profiles...")
        try:
            ehr_df = pd.read_csv(self.ehr_path)
        except FileNotFoundError:
            print("[!] Critical Error: EHR data not found. Run tabular_ehr_gen.py first.")
            return

        num_patients = len(ehr_df)
        print(f"[*] Generating {self.hours} hours of continuous wearable telemetry for {num_patients} nodes...")
        start_time = time.time()

        # Vectorized generation for extreme performance
        node_ids = np.repeat(ehr_df['Node_ID'].values, self.hours)
        timestamps = np.tile(np.arange(self.hours), num_patients)
        
        # Base vital logic tied to EHR
        # Higher BMI generally raises baseline HR
        base_hr = 65 + (ehr_df['BMI'].values - 20) * 0.5 
        base_spo2 = 98.0
        
        # Expand bases to match time-series shape
        base_hr_expanded = np.repeat(base_hr, self.hours)
        infected_status = np.repeat(ehr_df['Outbreak_Target'].values, self.hours)

        # 1. Circadian Rhythm (Sinusoidal wave: HR drops at night)
        circadian_wave = 5 * np.sin(2 * np.pi * (timestamps / 24) - (np.pi / 2))
        
        # 2. Add Stochastic Noise (Brownian motion proxy)
        hr_noise = self.np_rng.normal(0, 2, size=len(node_ids))
        spo2_noise = self.np_rng.normal(0, 0.5, size=len(node_ids))

        # 3. Pathogen Anomaly Injection (For infected patients only)
        # As time progresses (hours 0 to 168), the virus takes hold
        viral_progression = timestamps / self.hours
        
        hr_infection_spike = infected_status * (viral_progression * 15) # HR goes up by up to 15 bpm
        spo2_infection_drop = infected_status * (viral_progression * 6) # SpO2 drops by up to 6%
        
        # Calculate final vitals
        final_hr = base_hr_expanded + circadian_wave + hr_noise + hr_infection_spike
        final_spo2 = base_spo2 + spo2_noise - spo2_infection_drop
        
        # Cap biological limits
        final_spo2 = np.clip(final_spo2, 80.0, 100.0)

        # Build massive Dataframe
        telemetry_df = pd.DataFrame({
            "Node_ID": node_ids,
            "Hour": timestamps,
            "Heart_Rate": np.round(final_hr, 1),
            "SpO2": np.round(final_spo2, 1)
        })

        print(f"[*] Compressing and saving {len(telemetry_df):,} rows of telemetry data...")
        telemetry_df.to_csv(output_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"[+] Synthesis complete in {elapsed:.2f} seconds.")
        print(f"[+] Multi-modal dataset secured at: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ehr_file = os.path.join(current_dir, "synthetic_patients_ehr.csv")
    telemetry_file = os.path.join(current_dir, "wearable_telemetry.csv")
    
    generator = WearableTelemetryGenerator(ehr_path=ehr_file)
    generator.generate_telemetry(telemetry_file)