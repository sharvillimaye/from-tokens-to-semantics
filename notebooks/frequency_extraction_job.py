from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import Pool, get_context

from ngrams.streaming.pythia_streaming import (
    count_ngrams_at_checkpoints,
    canonical_pythia_steps,
    PYTHIA_TOKENS_PER_STEP,
)

# -----------------------------
# Config
# -----------------------------
SHARDS_GLOB = "./pile_deduped_all/document-*-of-*.bin"
TOKENIZER_NAME = "EleutherAI/pythia-70m"  # must match corpus
COUNT_MODE = "full"       # "full" (end-based) or "start" (start-based); keep "full" unless you matched start-based before
CHUNK_TOKENS = 3_000_000  # lower if RAM is tight; ~3M is ~ good on 32GB with small n-gram lengths
N_WORKERS = 2             # 1 = single-process; try 2–3 on fast NVMe, watch disk utilization
BUCKETS_Q = 4             # quantile buckets by step=1000 count
SAMPLES_PER_BUCKET = 25   # per bucket

# Avoid CPU over-subscription when using mp: (optional, harmless if unset)
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# -----------------------------
# Capital list (your snippet)
# -----------------------------
capital_pairs: List[Tuple[str, str]] = [
    ("United States", "Washington"), ("United Kingdom", "London"),
    ("France", "Paris"), ("Japan", "Tokyo"),
    ("Germany", "Berlin"), ("Canada", "Ottawa"),
    ("Italy", "Rome"), ("Spain", "Madrid"),
    ("South Korea", "Seoul"), ("Australia", "Canberra"),
    ("Mongolia", "Ulaanbaatar"), ("Kyrgyzstan", "Bishkek"),
    ("Tuvalu", "Funafuti"), ("Samoa", "Apia"),
    ("Afghanistan", "Kabul"), ("Albania", "Tirana"),
    ("Algeria", "Algiers"), ("Andorra", "Andorra la Vella"),
    ("Angola", "Luanda"), ("Antigua and Barbuda", "St. John's"),
    ("Argentina", "Buenos Aires"), ("Armenia", "Yerevan"),
    ("Austria", "Vienna"), ("Azerbaijan", "Baku"),
    ("Bahamas", "Nassau"), ("Bahrain", "Manama"),
    ("Bangladesh", "Dhaka"), ("Barbados", "Bridgetown"),
    ("Belarus", "Minsk"), ("Belgium", "Brussels"),
    ("Belize", "Belmopan"), ("Benin", "Porto-Novo"),
    ("Bhutan", "Thimphu"), ("Bolivia", "La Paz"),
    ("Bosnia and Herzegovina", "Sarajevo"), ("Botswana", "Gaborone"),
    ("Brazil", "Brasília"), ("Brunei", "Bandar Seri Begawan"),
    ("Bulgaria", "Sofia"), ("Burkina Faso", "Ouagadougou"),
    ("Burundi", "Gitega"), ("Cabo Verde", "Praia"),
    ("Cambodia", "Phnom Penh"), ("Cameroon", "Yaoundé"),
    ("Central African Republic", "Bangui"), ("Chad", "N'Djamena"),
    ("Chile", "Santiago"), ("China", "Beijing"),
    ("Colombia", "Bogotá"), ("Comoros", "Moroni"),
    ("Democratic Republic of the Congo", "Kinshasa"),
    ("Republic of the Congo", "Brazzaville"),
    ("Costa Rica", "San José"), ("Côte d'Ivoire", "Yamoussoukro"),
    ("Croatia", "Zagreb"), ("Cuba", "Havana"),
    ("Cyprus", "Nicosia"), ("Czechia", "Prague"),
    ("Denmark", "Copenhagen"), ("Djibouti", "Djibouti"),
    ("Dominica", "Roseau"), ("Dominican Republic", "Santo Domingo"),
    ("Ecuador", "Quito"), ("Egypt", "Cairo"),
    ("El Salvador", "San Salvador"), ("Equatorial Guinea", "Malabo"),
    ("Eritrea", "Asmara"), ("Estonia", "Tallinn"),
    ("Eswatini", "Mbabane"), ("Ethiopia", "Addis Ababa"),
    ("Fiji", "Suva"), ("Finland", "Helsinki"),
    ("Gabon", "Libreville"), ("Gambia", "Banjul"),
    ("Georgia", "Tbilisi"), ("Ghana", "Accra"),
    ("Greece", "Athens"), ("Grenada", "St. George's"),
    ("Guatemala", "Guatemala City"), ("Guinea", "Conakry"),
    ("Guinea-Bissau", "Bissau"), ("Guyana", "Georgetown"),
    ("Haiti", "Port-au-Prince"), ("Honduras", "Tegucigalpa"),
    ("Hungary", "Budapest"), ("Iceland", "Reykjavík"),
    ("India", "New Delhi"), ("Indonesia", "Jakarta"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Ireland", "Dublin"), ("Jamaica", "Kingston"),
    ("Jordan", "Amman"), ("Kazakhstan", "Astana"),
    ("Kenya", "Nairobi"), ("Kiribati", "Tarawa"),
    ("Kuwait", "Kuwait City"), ("Laos", "Vientiane"),
    ("Latvia", "Riga"), ("Lebanon", "Beirut"),
    ("Lesotho", "Maseru"), ("Liberia", "Monrovia"),
    ("Libya", "Tripoli"), ("Liechtenstein", "Vaduz"),
    ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg"),
    ("Madagascar", "Antananarivo"), ("Malawi", "Lilongwe"),
    ("Malaysia", "Kuala Lumpur"), ("Maldives", "Malé"),
    ("Mali", "Bamako"), ("Malta", "Valletta"),
    ("Marshall Islands", "Majuro"), ("Mauritania", "Nouakchott"),
    ("Mauritius", "Port Louis"), ("Mexico", "Mexico City"),
    ("Micronesia", "Palikir"), ("Moldova", "Chișinău"),
    ("Monaco", "Monaco"), ("Montenegro", "Podgorica"),
    ("Morocco", "Rabat"), ("Mozambique", "Maputo"),
    ("Myanmar", "Naypyidaw"), ("Namibia", "Windhoek"),
    ("Nauru", "Yaren"), ("Nepal", "Kathmandu"),
    ("Netherlands", "Amsterdam"), ("New Zealand", "Wellington"),
    ("Nicaragua", "Managua"), ("Niger", "Niamey"),
    ("Nigeria", "Abuja"), ("North Korea", "Pyongyang"),
    ("North Macedonia", "Skopje"), ("Norway", "Oslo"),
    ("Oman", "Muscat"), ("Pakistan", "Islamabad"),
    ("Palau", "Ngerulmud"), ("Panama", "Panama City"),
    ("Papua New Guinea", "Port Moresby"), ("Paraguay", "Asunción"),
    ("Peru", "Lima"), ("Philippines", "Manila"),
    ("Poland", "Warsaw"), ("Portugal", "Lisbon"),
    ("Qatar", "Doha"), ("Romania", "Bucharest"),
    ("Russia", "Moscow"), ("Rwanda", "Kigali"),
    ("Saint Kitts and Nevis", "Basseterre"),
    ("Saint Lucia", "Castries"),
    ("Saint Vincent and the Grenadines", "Kingstown"),
    ("San Marino", "San Marino"),
    ("São Tomé and Príncipe", "São Tomé"),
    ("Saudi Arabia", "Riyadh"), ("Senegal", "Dakar"),
    ("Serbia", "Belgrade"), ("Seychelles", "Victoria"),
    ("Sierra Leone", "Freetown"), ("Singapore", "Singapore"),
    ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"),
    ("Solomon Islands", "Honiara"), ("Somalia", "Mogadishu"),
    ("South Africa", "Pretoria"), ("South Sudan", "Juba"),
    ("Sri Lanka", "Sri Jayawardenepura Kotte"),
    ("Sudan", "Khartoum"), ("Suriname", "Paramaribo"),
    ("Sweden", "Stockholm"), ("Switzerland", "Bern"),
    ("Syria", "Damascus"), ("Taiwan", "Taipei"),
    ("Tajikistan", "Dushanbe"), ("Tanzania", "Dodoma"),
    ("Thailand", "Bangkok"), ("Timor-Leste", "Dili"),
    ("Togo", "Lomé"), ("Tonga", "Nukuʻalofa"),
    ("Trinidad and Tobago", "Port of Spain"),
    ("Tunisia", "Tunis"), ("Turkey", "Ankara"),
    ("Turkmenistan", "Ashgabat"), ("Uganda", "Kampala"),
    ("Ukraine", "Kyiv"), ("United Arab Emirates", "Abu Dhabi"),
    ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"),
    ("Vanuatu", "Port Vila"), ("Vatican City", "Vatican City"),
    ("Venezuela", "Caracas"), ("Vietnam", "Hanoi"),
    ("Yemen", "Sana'a"), ("Zambia", "Lusaka"),
    ("Zimbabwe", "Harare"),
]

def variants_for_pair(country: str, capital: str) -> List[str]:
    # Leading spaces are intentional (tokenizer behavior)
    return [
        f" {capital}",
        f" in {capital}",
        f" {capital},",
        f" capital of {country}",
    ]

# -----------------------------
# Helpers to encode/dedup grams
# -----------------------------
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def encode_capital_ngrams(strings: List[str], max_tokens: int = 3) -> Tuple[List[str], List[Tuple[int, ...]]]:
    texts, ids = [], []
    for s in strings:
        tok_ids = tok.encode(s, add_special_tokens=False)
        if 1 <= len(tok_ids) <= max_tokens:
            texts.append(s)
            ids.append(tuple(int(i) for i in tok_ids))
    # Dedup by token IDs to avoid duplicates that differ by Unicode accents/punctuation
    seen, out_texts, out_ids = set(), [], []
    for t, ng in zip(texts, ids):
        if ng not in seen:
            seen.add(ng); out_texts.append(t); out_ids.append(ng)
    return out_texts, out_ids

# Build the phrase pool
all_texts: List[str] = []
all_ids: List[Tuple[int, ...]] = []
for country, capital in capital_pairs:
    t, i = encode_capital_ngrams(variants_for_pair(country, capital), max_tokens=3)
    all_texts.extend(t); all_ids.extend(i)

# Keep a stable order and a mapping
P = len(all_texts)
assert P == len(all_ids), "mismatch"
phrase_to_row = {txt: r for r, txt in enumerate(all_texts)}

# -----------------------------
# Full checkpoint schedule
# -----------------------------
steps_all = canonical_pythia_steps(include_early=True)  # 154 steps
M = steps_all.size

# -----------------------------
# Parallel driver (phrase-sliced)
# -----------------------------
def _count_slice(args):
    """Worker: run streaming on a slice of phrases; returns (rows, counts[P_slice, M])."""
    (phrase_slice, steps, count_mode, chunk_tokens) = args
    res = count_ngrams_at_checkpoints(
        shards_glob=SHARDS_GLOB,
        phrases=phrase_slice,
        steps=steps.tolist(),
        tokenizer_name=TOKENIZER_NAME,
        chunk_tokens=chunk_tokens,
        tokens_per_step=PYTHIA_TOKENS_PER_STEP,
        count_mode=count_mode,
        keep_example_positions_per_pattern=0,
    )
    # Stack rows in the same order as phrase_slice
    counts = np.vstack([res[p]["count"] for p in phrase_slice]).astype(np.int64, copy=False)
    return (phrase_slice, counts)

def run_counts_parallel(phrases: List[str], steps: np.ndarray, n_workers: int) -> np.ndarray:
    """Split phrases across workers; merge stacked counts into [P, M]."""
    if n_workers <= 1 or len(phrases) == 0:
        # single-process fallback
        res = count_ngrams_at_checkpoints(
            shards_glob=SHARDS_GLOB,
            phrases=phrases,
            steps=steps.tolist(),
            tokenizer_name=TOKENIZER_NAME,
            chunk_tokens=CHUNK_TOKENS,
            tokens_per_step=PYTHIA_TOKENS_PER_STEP,
            count_mode=COUNT_MODE,
            keep_example_positions_per_pattern=0,
        )
        return np.vstack([res[p]["count"] for p in phrases]).astype(np.int64, copy=False)

    # Split phrases into ~equal buckets
    buckets: List[List[str]] = []
    per = math.ceil(len(phrases) / n_workers)
    for k in range(n_workers):
        buckets.append(phrases[k*per : (k+1)*per])

    tasks = [(bucket, steps, COUNT_MODE, CHUNK_TOKENS) for bucket in buckets if bucket]

    # Use spawn on some platforms to avoid forking large arrays
    with get_context("spawn").Pool(processes=len(tasks)) as pool:
        out = pool.map(_count_slice, tasks)

    # Merge back into [P, M] following original phrase order
    counts_full = np.zeros((len(phrases), steps.size), dtype=np.int64)
    for phrase_slice, counts in out:
        for local_idx, phrase in enumerate(phrase_slice):
            global_row = phrase_to_row[phrase]
            counts_full[global_row, :] = counts[local_idx, :]
    return counts_full

# -----------------------------
# Run
# -----------------------------
print(f"Counting {P} phrases across {M} checkpoints with N_WORKERS={N_WORKERS} (mode={COUNT_MODE})...")
counts_mat = run_counts_parallel(all_texts, steps_all, N_WORKERS)  # [P, M]
cutoffs = steps_all * np.int64(PYTHIA_TOKENS_PER_STEP)
freq_mat = counts_mat.astype(np.float64) / np.maximum(cutoffs[np.newaxis, :], 1.0)

# -----------------------------
# Bucket by frequency @ step=1000 and sample
# -----------------------------
# Locate step=1000 index
idx_1000 = int(np.where(steps_all == 1000)[0][0])
counts_step1000 = counts_mat[:, idx_1000]

def bucket_indices_by_quantiles(freq: np.ndarray, q: int = 4) -> List[np.ndarray]:
    if freq.size == 0:
        return []
    qs = np.quantile(freq, np.linspace(0, 1, q+1))
    bins = []
    for j in range(q):
        lo, hi = qs[j], qs[j+1]
        if j < q-1:
            idx = np.where((freq >= lo) & (freq <  hi))[0]
        else:
            idx = np.where((freq >= lo) & (freq <= hi))[0]
        bins.append(idx)
    return bins

bins = bucket_indices_by_quantiles(counts_step1000, q=BUCKETS_Q)
rng = np.random.default_rng(0)
selected_idx = np.concatenate([
    rng.choice(b, size=min(SAMPLES_PER_BUCKET, b.size), replace=False)
    for b in bins if b.size
]) if bins else np.array([], dtype=int)

# -----------------------------
# Materialize & save
# -----------------------------
rows = []
for j in selected_idx:
    rows.append({
        "text": all_texts[j],
        "ngram_ids": list(all_ids[j]),
        "len": len(all_ids[j]),
        "counts": counts_mat[j].tolist(),        # 154-length vector
        "freq": freq_mat[j].tolist(),            # normalized by tokens seen
        "count_step1000": int(counts_step1000[j]),
        "bucket": int(np.where([j in b for b in bins])[0][0]) if bins else -1,
    })
df_candidates = pd.DataFrame(rows).sort_values(["bucket", "count_step1000"]).reset_index(drop=True)

out_dir = Path("./exposure_artifacts/all_checkpoints_streaming")
out_dir.mkdir(parents=True, exist_ok=True)
df_candidates.to_parquet(out_dir / "capital_relation_candidates.parquet", index=False)
print(f"Saved {len(df_candidates)} candidates to {out_dir/'capital_relation_candidates.parquet'}")
print(df_candidates.head(min(20, len(df_candidates))))

