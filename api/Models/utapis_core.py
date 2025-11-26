import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

MODEL_PATH          = "xgb_manual_features_best.joblib"
FEATURE_SCHEMA_CSV  = "dataset_fitur_final.csv"
KAMUS_FILE          = "kamus(in)_clean.csv"
AKRONIM_FILE        = "daftar_akronim.csv"
ASING_FILE          = "nounlist_bersih.csv"

IGNORE_CANDIDATES = {"pernah"}
SATUAN_UMUM = set([
    'kilo','kilogram','kg','meter','m','km','lembar','orang','hari','jam','menit','detik',
    'rupiah','rp','persen','%','kepala','unit','liter','l','ml','bulan','tahun', 'siswa'
])
PRONOMINA_UMUM = set([
    'dia','ia','kamu','engkau','anda','saya','aku','kami','kita','mereka','beliau','ini','itu'
])
KONJUNGSI_PUN_BASE = {
    'ada','andai','atau','bagaimana','biar','kendati','meski','sekali','sungguh','walau','mau','betapa', 'kalau'
}
BAKU_KAH = {"apakah", "siapakah", "dimanakah", "kapankah", "mengapakah", "bagaimanakah"}
NEGATORS = r"(tidak|tak|bukan|belum|nggak|ga|enggak|nyaris\s+tidak|jarang)"
CONTRAST = r"(tapi|tetapi|namun|akan\s+tetapi)"
SCOPE_WINDOW = 8
MAIN_CONCESSIVE = r"(tetap|masih|ingin|akan|tetap\s+ingin|tetap\s+paham|tetap\s+saja)"
INCLUDE_KALAU = False
if INCLUDE_KALAU:
    KONJUNGSI_PUN_BASE = set(KONJUNGSI_PUN_BASE) | {'kalau'}

def should_ignore_candidate(cand: str, assets: Dict[str, Any]) -> bool:
    s = (cand or "").strip().lower()
    if s in IGNORE_CANDIDATES:
        return True

    if re.search(r"\b(pun|kah|lah)\b", s):
        return False
    if s.startswith("per "):
        return False

    pos_kamus = None
    if isinstance(assets, dict):
        pos_kamus = assets.get("pos_kamus", None)

    if isinstance(pos_kamus, dict):
        pos_info = pos_kamus.get(s)
        if isinstance(pos_info, dict):
            pass

    return False

def _tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def _join(tokens):
    return " ".join(t for t in tokens if t.strip())

import re

def normalize_sekali_pun_variants(text: str) -> str:
    """
    Normalisasi bentuk salah 'sekali-pun' → 'sekali pun' / 'sekalipun'
    berdasar konteks lokal (negasi / kontras).
    """
    t = text
    for m in re.finditer(r"\b[Ss]ekali[--]pun\b", t):
        start, end = m.span()
        tail = t[end:end+60].lower()
        head = t[max(0, start-40):start].lower()

        if re.search(r"\b(tidak|tak|bukan|belum|pernah|nyaris\s+tidak)\b", tail) or re.search(r"\b(tidak|tak|bukan|belum|pernah|nyaris\s+tidak)\b", head):
            correct = "sekali pun"
        elif start < 5 or re.search(r"[,;]\s*(namun|tapi|tetapi|akan\s+tetapi)", tail):
            correct = "sekalipun"
        else:
            correct = "sekali pun"

        t = t[:start] + correct + t[end:]
    return t

def _extract_features_for_sekalipun(text: str, pos_kamus: Dict[str, Dict[str,int]]|None=None) -> Dict[str, Any]:
    t = re.sub(r"\s+", " ", str(text).lower()).strip()
    toks = _tokenize(t)
    joined = _join(toks)

    m_sekalipun = re.search(r"\bsekalipun\b", joined)
    m_sekali_pun = re.search(r"\bsekali\s+pun\b", joined)
    idx_char = None
    target = None
    if m_sekalipun:
        idx_char = m_sekalipun.start()
        target = "sekalipun"
    elif m_sekali_pun:
        idx_char = m_sekali_pun.start()
        target = "sekali pun"

    start_tok = 0
    if idx_char is not None:
        start_tok = len(_join(_tokenize(joined[:idx_char])).split())

    toks_tail = toks[start_tok : start_tok + SCOPE_WINDOW + 1]
    tail = _join(toks_tail)

    feats = {
        "target_form": target,
        "has_negator_nearby": bool(re.search(rf"\b{NEGATORS}\b", tail)),
        "has_pernah_nearby": "pernah" in tail,
        "clause_initial": bool(re.search(r"(^|[.!?]\s+)(sekalipun|sekali\s+pun)", joined)),
        "comma_after_target": False,
        "has_contrast_later": False,
        "has_concessive_main": False,
        "pos_hint_sekali_NUM": False,
        "pos_hint_sekali_ADV": False,
        "pos_hint_pun_PART": False,
    }

    if idx_char is not None:
        after = joined[idx_char: idx_char + 80]
        feats["comma_after_target"] = "," in after
        feats["has_contrast_later"] = bool(re.search(rf"\b{CONTRAST}\b", joined[idx_char:]))
        
        tail_full = joined[idx_char:]
        parts = tail_full.split(",", 1)
        if len(parts) == 2:
            main_clause = parts[1]
            if re.search(rf"\b{MAIN_CONCESSIVE}\b", main_clause):
                feats["has_concessive_main"] = True

    if pos_kamus:
        if "sekali" in pos_kamus:
            feats["pos_hint_sekali_NUM"] = any(k.startswith("NUM") for k,v in pos_kamus["sekali"].items() if v)
            feats["pos_hint_sekali_ADV"] = any(k.startswith("ADV") for k,v in pos_kamus["sekali"].items() if v)
        if "pun" in pos_kamus:
            feats["pos_hint_pun_PART"] = any(k.startswith("PART") for k,v in pos_kamus["pun"].items() if v)

    return feats

def _score_sekalipun(feats: Dict[str, Any]) -> Dict[str, float]:
    s = {"sekali pun": 0.0, "sekalipun": 0.0}
    if feats["has_negator_nearby"]:
        s["sekali pun"] += 2.5
    if feats["has_pernah_nearby"]:
        s["sekali pun"] += 1.5
    if feats["clause_initial"] and feats["comma_after_target"]:
        s["sekalipun"] += 2.0
    if feats["has_contrast_later"]:
        s["sekalipun"] += 2.0
    if feats["has_concessive_main"]:
        s["sekalipun"] += 3.0
    if feats["pos_hint_pun_PART"]:
        s["sekali pun"] += 0.5
    if feats["pos_hint_sekali_NUM"]:
        s["sekali pun"] += 0.5
    if feats["pos_hint_sekali_ADV"]:
        s["sekalipun"] += 0.2
    if feats["target_form"] == "sekali pun":
        s["sekali pun"] += 0.3
    elif feats["target_form"] == "sekalipun":
        s["sekalipun"] += 0.3
    return s

def disambiguate_sekalipun(text: str, initial: str, pos_kamus: Dict[str, Dict[str,int]]|None=None) -> str:
    feats = _extract_features_for_sekalipun(text, pos_kamus)
    scores = _score_sekalipun(feats)
    if abs(scores["sekali pun"] - scores["sekalipun"]) < 0.25:
        return initial
    return max(scores, key=scores.get)

def build_pos_map(df_kamus: pd.DataFrame):
    pos_cols = [c for c in df_kamus.columns if c != 'word']
    pos_map = {}
    for _, row in df_kamus.iterrows():
        w = str(row['word']).lower()
        tags = {c.upper() for c in pos_cols if row[c] == 1}
        if tags:
            pos_map[w] = tags
    return pos_map

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', text.strip())
    return [p.strip() for p in parts if p.strip()]

def load_assets():
    assets = {}
    df_kamus = pd.read_csv(KAMUS_FILE)
    df_kamus['word'] = df_kamus['word'].astype(str).str.lower()
    assets['main_dictionary'] = set(df_kamus['word'])
    assets['pos_kamus'] = build_pos_map(df_kamus)

    df_akr = pd.read_csv(AKRONIM_FILE)
    col_akr = 'Akronim' if 'Akronim' in df_akr.columns else df_akr.columns[0]
    assets['acronyms'] = set(df_akr[col_akr].dropna().astype(str).str.lower())

    df_asing = pd.read_csv(ASING_FILE, header=None)
    assets['foreign_dictionary'] = set(df_asing[0].dropna().astype(str).str.lower())
    return assets

# ------- TOKENISASI & KANDIDAT -------

TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")

def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokenisasi ringan + posisi char (start, end)."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def extract_candidates(text: str, assets: Dict) -> List[Dict]:
    """
    Buat kandidat partikel (pun/lah/kah/per) dari teks.
    - Termasuk pola numeralia seperti 'satu per satu', 'tiga per empat'
    - Menghindari kandidat tumpang tindih (overlapping spans)
    """
    tokens = tokenize_with_spans(text)
    cands = []
    n = len(tokens)

    def add_cand(surface, i_center, span_start, span_end, token_before, token_after, token_after2=''):
        cands.append({
            'candidate': surface,
            'i_center': i_center,
            'span': (span_start, span_end),
            'token_before': token_before,
            'token_after': token_after,
            'token_after2': token_after2
        })

    for i, (tok, s, e) in enumerate(tokens):
        low = tok.lower()

        if low == 'per' and i > 0 and i + 1 < n:
            prev, nxt = tokens[i - 1][0].lower(), tokens[i + 1][0].lower()
            if prev.isdigit() or re.match(r"^(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|seratus|sejuta)$", prev):
                span_start, span_end = tokens[i - 1][1], tokens[i + 1][2]
                surface = f"{tokens[i - 1][0]} per {tokens[i + 1][0]}"
                add_cand(
                    surface, i, span_start, span_end,
                    tokens[i - 2][0] if i - 2 >= 0 else '',
                    tokens[i + 2][0] if i + 2 < n else ''
                )

        if low in {'pun', 'lah', 'kah'} and i > 0:
            w_prev, s_prev, e_prev = tokens[i - 1]
            surface = f"{w_prev} {tok}"
            add_cand(
                surface, i, s_prev, e,
                tokens[i - 2][0] if i - 2 >= 0 else '',
                tokens[i + 1][0] if i + 1 < n else '',
                tokens[i + 2][0] if i + 2 < n else ''
            )
            continue

        if (low.endswith('pun') or low.endswith('lah') or low.endswith('kah')) and not (
            i > 0 and tokens[i - 1][0].lower() in {'pun', 'lah', 'kah'}
        ):
            norm_tok = low.replace('-', '')
            if (
                norm_tok in assets['main_dictionary']
                or is_fused_kah_baku(tok, assets)
                or is_fused_lah_baku(tok, assets)
            ):
                continue

            add_cand(
                tok, i, s, e,
                tokens[i - 1][0] if i - 1 >= 0 else '',
                tokens[i + 1][0] if i + 1 < n else '',
                tokens[i + 2][0] if i + 2 < n else ''
            )

        if low == 'per' and i + 1 < n:
            nxt, s_n, e_n = tokens[i + 1]
            surface = f"{tok} {nxt}"
            add_cand(
                surface, i + 1, s, e_n,
                tokens[i - 1][0] if i - 1 >= 0 else '',
                tokens[i + 2][0] if i + 2 < n else '',
                tokens[i + 3][0] if i + 3 < n else ''
            )

        if low.startswith('per') and len(low) > 3:
            add_cand(
                tok, i, s, e,
                tokens[i - 1][0] if i - 1 >= 0 else '',
                tokens[i + 1][0] if i + 1 < n else '',
                tokens[i + 2][0] if i + 2 < n else ''
            )

    seen = set()
    uniq = []
    for c in cands:
        key = (c['span'], c['candidate'])
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    final_cands = []
    kept_spans = []
    for c in sorted(uniq, key=lambda x: (x['span'][0], x['span'][1])):
        s, e = c['span']
        overlap = any((s >= os and e <= oe) or (os >= s and oe <= e) for os, oe in kept_spans)
        if not overlap:
            final_cands.append(c)
            kept_spans.append((s, e))
    
    filtered = []
    for c in final_cands:
        cand = c['candidate'].lower()
        if ' ' not in cand and '-' not in cand and cand.endswith(('lah','kah','pun')):
            if cand in assets['main_dictionary']:
                continue
        filtered.append(c)

    return filtered

def split_affix(kata: str) -> str:
    k = str(kata).lower().replace(' ', '').replace('-', '')
    for suf in ('lah','kah','pun'):
        if k.endswith(suf):
            k = k[:-len(suf)]
            break
    return k

NUMERALIA = {'satu','dua','tiga','empat','lima','enam','tujuh','delapan','sembilan','sepuluh','sebelas'}
def is_per_preposition(kata_salah: str, token_after: str, assets: dict) -> bool:
    """
    True  → 'per' bermakna preposisi (tiap/demi/mulai) dan harus dipisah: 'per X'
    False → 'per' bukan preposisi (kemungkinan awalan per-): 'perX'
    Heuristik:
      - Jika 'per X' dan 'perX' (gabungan) ADA di kamus → BUKAN preposisi (awalan).
      - Jika X numeralia/satuan/digit → preposisi.
      - Jika X bertag BENDA/BILANGAN di kamus → preposisi.
      - Jika 'per X' dan 'perX' TIDAK ada di kamus → secara default anggap preposisi (lebih aman).
    """
    text = kata_salah.lower()
    if not re.search(r"\bper[\s\-]+", text):
        return False

    nxt = str(token_after).lower()
    if not nxt:
        return False

    main_dict = assets['main_dictionary']
    pos_tags = assets['pos_kamus'].get(nxt, set())

    if ('per' + nxt) in main_dict:
        return False

    if nxt in NUMERALIA or nxt in SATUAN_UMUM or nxt.isdigit():
        return True

    if any(t in {'KATA_BENDA', 'BENDA', 'KATA_BILANGAN', 'BILANGAN'} for t in pos_tags):
        return True

    return True

def is_per_prefix_form(kata_salah: str, main_dict: set) -> bool:
    """
    True  → 'per' adalah awalan pembentuk kata baku (contoh: 'perjalanan', 'perusahaan').
    False → 'per' bukan awalan; kemungkinan preposisi ('per orang', 'per tahun', 'per tiga'), dsb.
    Heuristik:
      1) Jika bentuk (tanpa hyphen, lower) ada di kamus → anggap valid awalan (kata baku).
      2) Jika 'per X' → valid awalan hanya jika 'perX' (gabung) ada di kamus. Selain itu anggap preposisi.
      3) Jika 'per...' (gabung) tapi tak ada di kamus:
         - Jika dasar = satuan umum / numeralia / digit → BUKAN awalan (preposisi).
         - Selain itu, asumsikan awalan (biar kata turunan baru yang belum terdaftar kamus tetap lolos).
    """
    ks = str(kata_salah).lower().replace('-', '').strip()
    parts = ks.split()

    if ks in main_dict:
        return True

    if len(parts) == 2 and parts[0] == 'per':
        joined = ''.join(parts)
        nxt = parts[1]
        if joined in main_dict:
            return True

        if nxt in SATUAN_UMUM or nxt in NUMERALIA or nxt.isdigit():
            return False

        return False

    if ks.startswith('per') and ks not in main_dict:
        base = ks[3:]
        if not base:
            return False

        if base in SATUAN_UMUM or base in NUMERALIA or base.isdigit():
            return False

        if base in main_dict:
            return True
        return True
    return False


def create_features_for_candidate(text: str, cand: Dict, assets: Dict) -> Dict:
    kata_salah = cand['candidate']
    token_before = cand['token_before'] or ''
    token_after  = cand['token_after'] or ''

    base_word_general = split_affix(kata_salah)

    tokens_ks = kata_salah.lower().split()
    first_tok = tokens_ks[0] if tokens_ks else ''
    base_for_pun = first_tok.replace('-', '')
    if base_for_pun.endswith('pun'):
        base_for_pun = base_for_pun[:-3]

    tags = assets['pos_kamus'].get(base_word_general, set())
    in_dict = base_word_general in assets['main_dictionary'] if base_word_general else False

    is_acr = (base_word_general.lower() in assets['acronyms']) if base_word_general else False
    is_for = (base_word_general.lower() in assets['foreign_dictionary']) if base_word_general else False

    per_prefix = is_per_prefix_form(kata_salah, assets['main_dictionary'])
    per_prep   = is_per_preposition(kata_salah, token_after, assets)

    ks_lower = kata_salah.lower()
    contains_per_token = bool(re.search(r"\bper\b", ks_lower))
    starts_with_per    = ks_lower.startswith("per")

    feats = {
        'len_error': len(kata_salah),
        'contains_space': int(' ' in kata_salah),
        'contains_hyphen': int('-' in kata_salah),
        'contains_pun': int('pun' in ks_lower),
        'contains_per': int(contains_per_token),
        'starts_with_per': int(starts_with_per),
        'contains_lah': int('lah' in ks_lower),
        'contains_kah': int('kah' in ks_lower),

        'is_after_per_a_unit': int(per_prep),
        'is_per_a_valid_prefix': int(per_prefix),

        'is_before_pun_a_pronoun': int(('pun' in ks_lower) and (
            token_before.lower() in PRONOMINA_UMUM or token_before.lower().endswith('nya'))
        ),
        'is_before_pun_a_conjunction': int(
            ('pun' in ks_lower) and (token_before.lower() in KONJUNGSI_PUN_BASE)
        ),
        'is_base_a_pun_conjunction': int(
            ('pun' in ks_lower) and (base_for_pun in KONJUNGSI_PUN_BASE)
        ),

        'is_base_uppercase': int(base_word_general.isupper() and len(base_word_general) > 1),
        'is_base_acronym': int(is_acr),
        'is_base_foreign': int(is_for),

        'base_word_pos_GANTI': int('KATA_GANTI' in tags or 'GANTI' in tags),
        'base_word_pos_BENDA': int('KATA_BENDA' in tags or 'BENDA' in tags),
        'base_word_pos_HUBUNG': int('KONJUNGSI' in tags or any(t.startswith('KATA_HUBUNG') or 'HUBUNG' in t for t in tags)),
        'base_word_pos_LAINNYA': int(len(tags) == 0),

        'is_not_in_main_dictionary': int((not in_dict) and base_word_general.isalpha()),
        'digit_count': sum(c.isdigit() for c in base_word_general),
        # kontek opsional: token_before/after literal bisa disimpan jika ingin debug
    }
    return feats

def is_already_correct(surface: str) -> bool:
    """
    Deteksi bentuk yang sangat mungkin SUDAH baku (jangan dikoreksi):
      - Pola pecahan/urutan numeralia: 'satu per satu', 'tiga per empat', dst.
      - 'per' + satuan umum/numeralia/digit: 'per orang', 'per jam', 'per 100'
      - Konjungsi difusi + partikel: 'walaupun', 'meskipun', 'kalaupun', dsb.
      - Bentuk tanya baku ber-'-kah': 'apakah', 'siapakah', dll. (pakai BAKU_KAH)
      - Akronim + -kah/-pun dengan hyphen (mis. 'HUMAS-kah', 'KPU-pun') dianggap valid di layer lain.
    Catatan: fungsi ini tidak akses kamus; guard kamus ada di correction_for_rule.
    """
    low = surface.lower().strip()
    low_sp = low.replace('-', ' ')

    if re.search(r"\b(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas)\s+per\s+"
                 r"(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas)\b", low_sp):
        return True

    m = re.search(r"\bper[\s\-]+([a-z0-9%]+)\b", low_sp)
    if m:
        nxt = m.group(1)
        if nxt in SATUAN_UMUM or nxt in NUMERALIA or nxt.isdigit():
            return True

    if any(low == f"{k}{suf}" for k in KONJUNGSI_PUN_BASE for suf in ('pun', 'lah', 'kah')):
        return True

    if low.replace('-', '') in BAKU_KAH:
        return True

    return False

def is_fused_kah_baku(surface: str, assets: Dict) -> bool:
    s = surface.strip()
    low = s.lower()
    if ' ' in s or '-' in s or not low.endswith('kah'):
        return False
    base = low[:-3]
    if not base:
        return False
    return (base in assets['main_dictionary']
            and base not in assets['acronyms']
            and base not in assets['foreign_dictionary'])

def is_fused_lah_baku(surface: str, assets: Dict) -> bool:
    s = surface.strip()
    low = s.lower()
    if ' ' in s or '-' in s or not low.endswith('lah'):
        return False
    base = low[:-3]
    if not base:
        return False
    return (base in assets['main_dictionary']
            and base not in assets['acronyms']
            and base not in assets['foreign_dictionary'])


def correction_for_rule(label: int, surface: str, assets: Dict) -> str:
    s = surface.strip()
    low = s.lower()
    norm_no_hyphen = s.lower().replace('-', '')
    if ' ' not in s and norm_no_hyphen in assets['main_dictionary']:
        return s

    if is_fused_kah_baku(s, assets):
        return s
    if is_fused_lah_baku(s, assets):
        return s

    m = re.match(r"^(.*?)\b([A-Za-z0-9\-]+)\s+(kah|lah)$", s, flags=re.IGNORECASE)
    if m:
        prefix, base_tok, particle = m.group(1), m.group(2), m.group(3).lower()
        base_clean = base_tok.strip()
        base_norm  = base_clean.replace(' ', '').replace('-', '').lower()
        is_acronym = base_clean.isupper()
        is_foreign = (base_norm in assets['foreign_dictionary']) and (base_norm not in assets['main_dictionary'])

        if is_acronym or is_foreign:
            return f"{prefix}{base_clean}-{particle}"
        else:
            return f"{prefix}{base_clean}{particle}"
        
    m2 = re.match(r"^(.*?)\b([A-Za-z0-9\-]+)\s+pun$", s, flags=re.IGNORECASE)
    if m2:
        prefix, base_tok = m2.group(1), m2.group(2)
        base_clean = base_tok.strip()
        base_norm  = base_clean.replace(' ', '').replace('-', '').lower()
        is_acronym = base_clean.isupper()
        is_foreign = (base_norm in assets['foreign_dictionary']) and (base_norm not in assets['main_dictionary'])
        if is_acronym or is_foreign:
            return f"{prefix}{base_clean}-pun"

        if base_norm in KONJUNGSI_PUN_BASE:
            return f"{prefix}{base_clean}pun"
        return f"{prefix}{base_clean} pun"

    if is_already_correct(s):
        return s
    
    if surface.lower().replace('-', '') in BAKU_KAH:
        return surface
    
    if re.search(r"(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\s+per\s+(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)", low):
        return s

    def base_and_particle(s):
        if ' ' in s:
            parts = s.split()
            if parts[-1].lower() in ('pun','lah','kah'):
                return ' '.join(parts[:-1]), parts[-1].lower()
        if '-' in s:
            for suf in ('-pun','-lah','-kah'):
                if low.endswith(suf):
                    return s[:-len(suf)], suf[1:]
        for suf in ('pun','lah','kah'):
            if low.endswith(suf):
                return s[:-len(suf)], suf
        return s, ''

    base_asli, part = base_and_particle(s)
    base_stripped = base_asli.replace('-', '').strip()

    if label == 1:
        if base_asli.lower().strip() in KONJUNGSI_PUN_BASE:
            return base_asli.replace(' ', '') + 'pun'
        return s

    if label == 2:
        if ' ' in s and s.lower().strip().endswith(' pun'):
            return s
        return base_asli.replace('-', '').strip() + ' ' + 'pun'

    norm = s.lower().replace(' ', '').replace('-', '')
    if norm in assets['main_dictionary']:
        return s

    if label == 3:
        if ' ' in s:
            parts = s.split()
            return f"{parts[0].lower()} {parts[1]}"
        if low.startswith('per'):
            return 'per ' + s[3:]
        return s 
    if label == 4:
        parts = s.split()
        if len(parts) >= 2 and parts[0].lower() == 'per':
            nxt = parts[1]
            nxt_low = nxt.lower()
            is_prep_like = (
                (nxt_low in NUMERALIA) or
                (nxt_low in SATUAN_UMUM) or
                nxt_low.isdigit() or
                any(t in {'KATA_BENDA','BENDA','KATA_BILANGAN','BILANGAN'} for t in assets['pos_kamus'].get(nxt_low, set())) or
                (('per' + nxt_low) not in assets['main_dictionary'])
            )
            if is_prep_like:
                return f"per {nxt}"
            else:
                return f"per{nxt}"
        return s.replace(' ', '')

    if label == 5:
        if part in ('lah','kah'):
            return base_asli.replace(' ', '') + part
        for suf in ('lah','kah'):
            if low.endswith(suf):
                return base_asli.replace(' ', '').replace('-', '') + suf
        return s
    if label == 6:
        return base_stripped + '-pun'
    if label == 7:
        if part in ('lah','kah'):
            return base_stripped + f"-{part}"
        for suf in ('lah','kah'):
            if low.endswith(suf):
                return base_stripped + f"-{suf}"
        return base_stripped
    if label == 8:
        return base_stripped + '-pun'
    if label == 9:
        if part in ('lah','kah'):
            return base_stripped + f"-{part}"
        for suf in ('lah','kah'):
            if low.endswith(suf):
                return base_stripped + f"-{suf}"
        return base_stripped

    return s

def load_feature_schema():
    """
    Ambil urutan/daftar kolom fitur dari dataset_fitur_final.csv (training).
    Kita buang kolom 'label'. Kolom di inference diselaraskan ke sini.
    """
    df_schema = pd.read_csv(FEATURE_SCHEMA_CSV, nrows=1)
    cols = [c for c in df_schema.columns if c != 'label']
    return cols

def align_features_df(feat_rows: List[Dict], expected_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(feat_rows)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        df = df.drop(columns=extra)
    df = df[expected_cols]
    for c in df.columns:
        if df[c].dtype == 'bool':
            df[c] = df[c].astype(np.int8)
        elif str(df[c].dtype).startswith('int'):
            df[c] = df[c].astype(np.int16)
        elif str(df[c].dtype).startswith('float'):
            df[c] = df[c].astype(np.float32)
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.int16)
    return df.astype(np.float32)

def predict_sentence(text: str) -> List[Dict]:
    model = joblib.load(MODEL_PATH)
    assets = load_assets()
    expected_cols = load_feature_schema()
    text = normalize_sekali_pun_variants(text)
    sentences = split_sentences(text)
    raw_cands = extract_candidates(text, assets)
    if not raw_cands:
        return []

    cands = [c for c in raw_cands if not should_ignore_candidate(c.get('candidate', ''), assets)]
    if not cands:
        return []

    feat_rows = []
    for c in cands:
        feats = create_features_for_candidate(text, c, assets)
        feat_rows.append(feats)

    X = align_features_df(feat_rows, expected_cols)

    preds = model.predict(X)
    proba = None
    try:
        proba = model.predict_proba(X)
    except Exception:
        pass

    results = []
    for i, c in enumerate(cands):
        label_i = int(preds[i])
        p_i = float(np.max(proba[i])) if proba is not None else None

        corr = correction_for_rule(label_i, c['candidate'], assets)
        cand_l = str(c.get('candidate', '')).lower()
        corr_l = str(corr or "").lower()
        needs_disambig = (
            re.search(r"\b(sekalipun|sekali\s+pun|sekali[-–]pun)\b", cand_l)
            or re.search(r"\b(sekalipun|sekali\s+pun|sekali[-–]pun)\b", corr_l)
        )

        if needs_disambig:
            pos_kamus = assets.get("pos_kamus") if isinstance(assets, dict) else None
            corr = disambiguate_sekalipun(text, corr, pos_kamus=pos_kamus)

        def _norm(x: str) -> str:
            return (x or "").strip().lower()

        is_correct = (_norm(corr) == _norm(c['candidate']))

        entry = {
            'candidate': c['candidate'],
            'span': c['span'],
            'token_before': c['token_before'],
            'token_after': c['token_after'],
            'status': 'correct' if is_correct else 'incorrect',
        }

        if not is_correct:
            entry['correction'] = corr
            entry['pred_label'] = label_i
            if p_i is not None:
                entry['prob'] = p_i

        results.append(entry)

    return results

if __name__ == "__main__":
    text = (
        "Guru lah yang paling awal tiba di kelas, sedangkan kepala sekolah pun datang tidak lama kemudian. "
        "Apakah murid-murid itu merasa lelah kah setelah mengikuti upacara panjang tadi pagi? "
        "Namun perjam pelajaran hanya diberikan selama empat puluh menit. "
        "Siswa per daerah wajib membawa buku catatan sendiri-sendiri, "
        "dan biaya pendaftaran per siswa sudah ditentukan oleh pihak sekolah. "
        "Walaupun aturan itu terasa ketat, masyarakat pun memahami pentingnya disiplin. "
        "Di sisi lain, media pun menyoroti kebijakan baru tersebut. "
        "sekalipun tidak pernah ke Amerika, saya tahu jalan di Amerika. "
        "Update-pun diterbitkan setiap pagi oleh tim IT, sedangkan HUMAS-kah yang akan memberikan klarifikasi resmi. "
        "DPR lah yang akhirnya menyetujui rancangan tersebut setelah rapat panjang. "
        "Perjalanan per tahun diwajibkan melaporkan laporan keuangannya. "
        "KPU pun berjanji akan memperketat sistem keamanan data. "
        "Tuhan lah berkata 'semua manusia itu baik'"
        "Benarkah semua pihak telah sepakat? Presidenpun mengonfirmasi hal tersebut satu per satu."
    )
    # debug_inspect_candidate(text, target='Benarkah')
    out = predict_sentence(text)
    print(f"Input: {text}\n")
    print("Deteksi kandidat:")
    for item in out:
        print(item)