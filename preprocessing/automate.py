import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Pisahkan fitur dan target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gabungkan kembali
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["target"] = y

    # Simpan hasil preprocessing
    df_processed.to_csv(output_path, index=False)

    print("✅ Preprocessing selesai!")

if __name__ == "__main__":
    # otomatis ambil path relatif dari lokasi file ini
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(base_dir, "..", "dataset_raw", "heart.csv")
    output_file = os.path.join(base_dir, "..", "dataset_preprocessing.csv")

    preprocess_data(input_file, output_file)