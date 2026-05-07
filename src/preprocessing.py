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
    "Usia": {"Usia Produktif": 0, "Usia Lanjut": 1},
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
        self.numerical_cols = ["Ket.Usia", "BB", "TB", "IMT"]
        self.categorical_cols = [
            "Usia",
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
        - Menghitung IMT dari BB dan TB jika belum ada / perlu dihitung ulang
          IMT = BB / (TB/100)^2

        CATATAN KONSISTENSI:
        BB & TB di-cast ke integer (mengikuti perilaku notebook disertasi)
        agar hasil IMT 100% bit-identical antara web (Laravel+Flask) dan notebook.
        Laravel menyimpan BB/TB sebagai FLOAT di MySQL, namun input medis selalu
        bilangan bulat (kg / cm), sehingga koersi ke int aman & deterministik.
        """
        df_fe = df.copy()

        if "BB" in df_fe.columns and "TB" in df_fe.columns:
            # Konversi ke numeric dulu (handle string/None)
            df_fe["BB"] = pd.to_numeric(df_fe["BB"], errors="coerce")
            df_fe["TB"] = pd.to_numeric(df_fe["TB"], errors="coerce")
            # Drop baris dengan BB/TB NaN sebelum cast ke int
            df_fe = df_fe.dropna(subset=["BB", "TB"]).reset_index(drop=True)
            # Cast ke int untuk konsistensi dengan notebook (astype(int))
            df_fe["BB"] = df_fe["BB"].astype(int)
            df_fe["TB"] = df_fe["TB"].astype(int)
            # Hitung IMT: BB / (TB dalam meter)^2
            tb_meter = df_fe["TB"] / 100
            df_fe["IMT"] = df_fe["BB"] / (tb_meter**2)
            df_fe["IMT"] = df_fe["IMT"].round(2)
            print(
                f"Feature Engineering: IMT dihitung dari BB dan TB ({len(df_fe)} records)"
            )

        return df_fe

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
            "usia": "Usia",
            "ket_usia": "Ket.Usia",
            "jenis_kelamin": "Jenis Kelamin",
            "status_bekerja": "Status Bekerja",
            "bb": "BB",
            "tb": "TB",
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

            # Konversi numerik
            if mapped_key in ["Ket.Usia", "BB", "TB"]:
                try:
                    encoded_data[mapped_key] = float(value)
                except (ValueError, TypeError):
                    encoded_data[mapped_key] = 0
            # Encode kategorikal
            elif mapped_key in self.label_encoders:
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

        # Feature Engineering: Hitung IMT dari BB dan TB
        if "BB" in encoded_data and "TB" in encoded_data:
            bb = encoded_data["BB"]
            tb = encoded_data["TB"]
            if tb > 0:
                tb_meter = tb / 100
                encoded_data["IMT"] = round(bb / (tb_meter**2), 2)
            else:
                encoded_data["IMT"] = 0
            print(
                f"Feature Engineering: IMT = {encoded_data['IMT']} (BB={bb}, TB={tb})"
            )

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
        ("Usia", "Usia Produktif"),
        ("Jenis Kelamin", "Laki-Laki"),
        ("Status Gizi", "Gizi Normal"),
        ("Keberhasilan Pengobatan", "Berhasil"),
    ]
    for col, value in test_cases:
        encoded = preprocessor.encode_value(col, value)
        decoded = preprocessor.decode_value(col, encoded)
        print(f"{col}: '{value}' -> {encoded} -> '{decoded}'")
