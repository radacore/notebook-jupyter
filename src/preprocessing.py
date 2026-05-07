"""
Data Preprocessing Module untuk Prediksi Keberhasilan Pengobatan MDR-TB

Modul ini menangani:
- Data cleaning
- Outlier detection menggunakan metode IQR
- Label encoding untuk data kategorikal (Manual Mapping)
"""

from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Manual Label Encoding Mappings (Hardcoded untuk konsistensi)
LABEL_ENCODINGS = {
    "Kategori Usia": {"Usia Produktif": 0, "Usia Lanjut": 1},
    "Jenis Kelamin": {"Laki-Laki": 0, "Laki-laki": 0, "Perempuan": 1},
    "Status Bekerja": {"Tidak Bekerja": 0, "Bekerja": 1},
    "Status Gizi": {"Gizi Kurang": 0, "Gizi Normal": 1, "Gizi Lebih": 2},
    "Status Merokok": {"Tidak Merokok": 0, "Merokok": 1},
    "Pemeriksaan Kontak": {"Tidak": 0, "Ya": 1, "Ada": 1},
    "Riwayat_DM": {"Tidak": 0, "Ada": 1, "Ya": 1},
    "Riwayat_HIV": {"Tidak": 0, "Ada": 1, "Ya": 1},
    "Komorbiditas": {"Tidak Ada": 0, "Tidak": 0, "Ada": 1, "Ya": 1},
    "Kepatuhan Minum Obat": {"Patuh": 0, "Tidak Patuh": 1, "Kurang Patuh": 1},
    "Efek Samping Obat": {"Tidak Ada Keluhan": 0, "Ada Keluhan": 1},
    "Riwayat Pengobatan Sebelumnya": {
        "Baru": 0,
        "Kasus Baru": 0,
        "Pengobatan Ulang": 1,
        "Kasus Lama": 1,
    },
    "Panduan Pengobatan": {"Jangka Pendek": 0, "Jangka Panjang": 1},
    "Keberhasilan Pengobatan": {"Berhasil": 0, "Tidak Berhasil": 1},
}

# Reverse mappings untuk decode
LABEL_DECODINGS = {
    col: {v: k for k, v in mappings.items()}
    for col, mappings in LABEL_ENCODINGS.items()
}


class DataPreprocessor:
    """Kelas untuk melakukan preprocessing data MDR-TB"""

    def __init__(self):
        self.label_encoders = LABEL_ENCODINGS
        self.label_decoders = LABEL_DECODINGS
        self.numerical_cols = []
        self.categorical_cols = [
            "Kategori Usia",
            "Jenis Kelamin",
            "Status Bekerja",
            "Status Gizi",
            "Status Merokok",
            "Pemeriksaan Kontak",
            "Riwayat_DM",
            "Riwayat_HIV",
            "Komorbiditas",
            "Kepatuhan Minum Obat",
            "Efek Samping Obat",
            "Riwayat Pengobatan Sebelumnya",
            "Panduan Pengobatan",
        ]
        self.target_col = "Keberhasilan Pengobatan"
        self.feature_cols = self.numerical_cols + self.categorical_cols

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data dari file CSV"""
        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Membersihkan data:
        - Hapus kolom yang tidak diperlukan
        - Handle missing values
        """
        df_clean = df.copy()

        # Hapus kolom 'Efek_Samping_Obat' yang memiliki banyak missing values
        if "Efek_Samping_Obat" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Efek_Samping_Obat"])

        # Hapus baris dengan missing values
        df_clean = df_clean.dropna()

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        return df_clean

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering:
        - Saat ini tidak ada feature engineering tambahan.
        - Fitur numerik (BB, TB, IMT, Ket.Usia) sudah dihapus
          agar konsisten dengan fitur yang digunakan di web.
        """
        return df.copy()

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghapus outliers menggunakan metode IQR
        """
        df_clean = df.copy()

        for col in self.numerical_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]

        return df_clean.reset_index(drop=True)

    def encode_value(self, column: str, value: str) -> int:
        """Encode single value menggunakan manual mapping"""
        if column in self.label_encoders:
            mapping = self.label_encoders[column]
            # Coba exact match dulu
            if value in mapping:
                return mapping[value]
            # Coba case-insensitive match
            value_lower = str(value).lower().strip()
            for key, encoded in mapping.items():
                if key.lower().strip() == value_lower:
                    return encoded
            # Default: return -1 jika tidak ditemukan
            print(f"Warning: Unknown value '{value}' for column '{column}'")
            return -1
        return -1

    def decode_value(self, column: str, encoded: int) -> str:
        """Decode single value menggunakan manual mapping"""
        if column in self.label_decoders:
            mapping = self.label_decoders[column]
            return mapping.get(encoded, "Unknown")
        return "Unknown"

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Melakukan label encoding pada kolom kategorikal menggunakan manual mapping

        Args:
            df: DataFrame untuk diproses
            fit: Parameter ini diabaikan karena menggunakan manual mapping
        """
        df_encoded = df.copy()

        # Encode semua kolom kategorikal + target
        cols_to_encode = self.categorical_cols + [self.target_col]

        for col in cols_to_encode:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: self.encode_value(col, str(x))
                )

        return df_encoded

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Pipeline preprocessing lengkap:
        1. Cleaning Data
        2. Feature Engineering (hitung IMT)
        3. Outlier Detection (IQR)
        4. Label Encoding
        """
        df_processed = self.clean_data(df)
        df_processed = self.feature_engineering(df_processed)
        df_processed = self.remove_outliers(df_processed)
        df_processed = self.encode_categorical(df_processed, fit=fit)
        return df_processed

    def get_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Memisahkan features dan target dari DataFrame
        """
        # Pilih hanya kolom yang ada di dataset
        available_features = [col for col in self.feature_cols if col in df.columns]
        X = df[available_features]
        y = df[self.target_col]
        return X, y

    def preprocess_single_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Preprocess satu input data untuk prediksi.
        Termasuk Feature Engineering: menghitung IMT dari BB dan TB.
        """
        # Mapping dari snake_case (frontend) ke format asli
        field_mapping = {
            "kategori_usia": "Kategori Usia",
            "jenis_kelamin": "Jenis Kelamin",
            "status_bekerja": "Status Bekerja",
            "status_gizi": "Status Gizi",
            "status_merokok": "Status Merokok",
            "pemeriksaan_kontak": "Pemeriksaan Kontak",
            "riwayat_dm": "Riwayat_DM",
            "riwayat_hiv": "Riwayat_HIV",
            "komorbiditas": "Komorbiditas",
            "kepatuhan_minum_obat": "Kepatuhan Minum Obat",
            "efek_samping_obat": "Efek Samping Obat",
            "riwayat_pengobatan": "Riwayat Pengobatan Sebelumnya",
            "panduan_pengobatan": "Panduan Pengobatan",
        }

        # Convert input keys dan encode
        encoded_data = {}
        for key, value in input_data.items():
            mapped_key = field_mapping.get(key, key)

            if mapped_key in self.label_encoders:
                if isinstance(value, int) or (
                    isinstance(value, str) and value.isdigit()
                ):
                    int_val = int(value)
                    if int_val in self.label_encoders[mapped_key].values():
                        encoded_data[mapped_key] = int_val
                    else:
                        encoded_data[mapped_key] = self.encode_value(
                            mapped_key, str(value)
                        )
                else:
                    encoded_data[mapped_key] = self.encode_value(mapped_key, str(value))
            else:
                encoded_data[mapped_key] = value

        df = pd.DataFrame([encoded_data])

        # Pilih hanya kolom fitur
        available_features = [col for col in self.feature_cols if col in df.columns]
        return df[available_features]

    def save(self, filepath: str):
        """Simpan preprocessor ke file"""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "DataPreprocessor":
        """Load preprocessor dari file"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test encoding
    preprocessor = DataPreprocessor()

    print("=== Manual Label Encoding Mappings ===")
    for col, mapping in LABEL_ENCODINGS.items():
        print(f"\n{col}:")
        for value, encoded in mapping.items():
            print(f"  {value} -> {encoded}")

    # Test single encoding
    print("\n=== Test Single Encoding ===")
    test_cases = [
        ("Kategori Usia", "Usia Produktif"),
        ("Jenis Kelamin", "Laki-Laki"),
        ("Status Gizi", "Gizi Normal"),
        ("Keberhasilan Pengobatan", "Berhasil"),
    ]
    for col, value in test_cases:
        encoded = preprocessor.encode_value(col, value)
        decoded = preprocessor.decode_value(col, encoded)
        print(f"{col}: '{value}' -> {encoded} -> '{decoded}'")
