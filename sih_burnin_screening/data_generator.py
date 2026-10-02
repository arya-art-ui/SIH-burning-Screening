import numpy as np
import pandas as pd


def generate_ess_dataset(num_components=1500, random_seed=42):
    """Generate synthetic ESS burn-in data for 0h, 24h and 168h."""
    rng = np.random.default_rng(random_seed)

    lots = [f"LOT_{i:02d}" for i in range(1, 11)]
    data = []

    for i in range(1, num_components + 1):
        component_id = f"CMP-{i:04d}"
        lot_id = rng.choice(lots)
        is_defective = rng.random() < 0.10

        # 0h baseline
        v_0h = rng.normal(5.0, 0.05)
        i_0h = rng.normal(120.0, 2.0)
        temp_0h = rng.normal(35.0, 1.5)

        if not is_defective:
            # Normal degradation
            v_24h = v_0h + rng.normal(0.01, 0.01)
            i_24h = i_0h + rng.normal(0.5, 0.2)
            temp_24h = temp_0h + rng.normal(1.0, 0.5)

            v_96h = v_24h + rng.normal(0.01, 0.015)
            i_96h = i_24h + rng.normal(0.5, 0.3)
            temp_96h = temp_24h + rng.normal(0.8, 0.5)

            v_168h = v_24h + rng.normal(0.02, 0.02)
            i_168h = i_24h + rng.normal(1.0, 0.5)
            temp_168h = temp_24h + rng.normal(2.0, 0.8)
            failed_at_168h = 0
        else:
            # Abnormal early drift
            v_24h = v_0h + rng.normal(-0.15, 0.05)
            i_24h = i_0h + rng.normal(8.0, 2.0)
            temp_24h = temp_0h + rng.normal(6.0, 1.5)

            v_96h = v_24h - rng.uniform(0.1, 0.4)
            i_96h = i_24h + rng.uniform(8.0, 20.0)
            temp_96h = temp_24h + rng.uniform(6.0, 15.0)

            v_168h = v_24h - rng.uniform(0.3, 0.8)
            i_168h = i_24h + rng.uniform(20.0, 50.0)
            temp_168h = temp_24h + rng.uniform(15.0, 30.0)
            failed_at_168h = 1

        data.append({
            "component_id": component_id,
            "lot_id": lot_id,
            "v_0h": v_0h,
            "i_0h": i_0h,
            "temp_0h": temp_0h,
            "v_24h": v_24h,
            "i_24h": i_24h,
            "temp_24h": temp_24h,
            "v_96h": v_96h,
            "i_96h": i_96h,
            "temp_96h": temp_96h,
            "v_168h": v_168h,
            "i_168h": i_168h,
            "temp_168h": temp_168h,
            "actual_failure_168h": failed_at_168h,
        })

    df = pd.DataFrame(data)

    # Early 0h -> 24h drift features
    df["delta_v_0_24"] = df["v_24h"] - df["v_0h"]
    df["delta_i_0_24"] = df["i_24h"] - df["i_0h"]
    df["delta_temp_0_24"] = df["temp_24h"] - df["temp_0h"]

    return df


if __name__ == "__main__":
    df = generate_ess_dataset()
    output_file = "ESS_predictive_screening_dataset_1500.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset generated successfully: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
