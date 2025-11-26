import os
from typing import List, Dict, Any
from . import utapis_core as core
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
core.MODEL_PATH          = os.path.join(DATA_DIR, "xgb_manual_features_best.joblib")
core.FEATURE_SCHEMA_CSV  = os.path.join(DATA_DIR, "dataset_fitur_final.csv")
core.KAMUS_FILE          = os.path.join(DATA_DIR, "kamus(in)_clean.csv")
core.AKRONIM_FILE        = os.path.join(DATA_DIR, "daftar_akronim.csv")
core.ASING_FILE          = os.path.join(DATA_DIR, "nounlist_bersih.csv")


class UtapisDetector:
    """
    Wrapper simpel untuk model UTAPIS supaya gampang dipanggil dari Flask.

    Cara pakai (di app.py):
        from Models.utapis_detector import UtapisDetector

        detector = UtapisDetector()
        result = detector.detect_particles(paragraph)
    """

    def __init__(self) -> None:
        pass

    def detect_particles(self, text: str) -> List[Dict[str, Any]]:
        """
        Jalankan deteksi kesalahan partikel pada satu paragraf teks.

        Output: list of dict, misalnya:
        {
          "candidate": "diapun",
          "span": (start_idx, end_idx),
          "token_before": "dia",
          "token_after": "",
          "status": "error" / "correct",
          "correction": "dia pun",
          "pred_label": 2,
          "prob": 0.93
        }
        """
        if not isinstance(text, str):
            text = str(text or "")

        text = text.strip()
        if not text:
            return []

        results = core.predict_sentence(text)
        return results
