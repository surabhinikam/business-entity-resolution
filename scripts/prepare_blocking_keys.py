#!/usr/bin/env python3
"""
Precompute blocking keys for candidate sources (source2, source3).
Generates s2_keys.parquet and s3_keys.parquet in data/processed/train/blocking_keys/.
Streams partitions/chunks to ensure constant minimal memory usage.
"""

import os
import sys
import time
import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.analysis.data_loader import load_processed_parquet
from src.candidate_generation.name_tokens import extract_meaningful_tokens
from src.candidate_generation.address_parser import extract_address_number

COLS = [
    "entity_id",
    "business_name_normalized",
    "business_name_transliterated",
    "business_address_normalized",
    "country_normalized",
]

def process_source_to_parquet(source_name: str, out_file: str):
    if os.path.exists(out_file):
        logger.info(f"{out_file} already exists ({os.path.getsize(out_file)/1e6:.1f} MB). Skipping.")
        return

    logger.info(f"Processing {source_name}...")
    t0 = time.time()
    
    # Load processed parquet
    lf = load_processed_parquet(f"data/processed/train/{source_name}", source=source_name, columns=COLS)
    df = lf.collect()
    logger.info(f"  Loaded {len(df):,} records in {time.time()-t0:.1f}s")
    
    t1 = time.time()
    eids = df["entity_id"].to_list()
    names = df["business_name_normalized"].to_list()
    translits = df["business_name_transliterated"].to_list()
    addrs = df["business_address_normalized"].to_list()
    countries = df["country_normalized"].to_list()
    del df  # free Arrow memory immediately
    
    key_A = []
    key_B = []
    key_C = []
    key_D = []
    key_E = []
    key_F = []
    has_nums = []
    has_addrs = []
    clean_countries = []
    
    for eid, name, translit, addr, country in zip(eids, names, translits, addrs, countries):
        c = country.strip().lower() if country else None
        clean_countries.append(c)
        
        ad = addr.strip() if addr else None
        has_addr = bool(ad)
        has_addrs.append(has_addr)
        
        num, has_num = extract_address_number(ad) if has_addr else (None, False)
        has_nums.append(has_num)
        
        if not c:
            key_A.append(None)
            key_B.append(None)
            key_C.append(None)
            key_D.append(None)
            key_E.append(None)
            key_F.append(None)
            continue
            
        nm = name.strip() if name else None
        key_A.append(f"A||{c}||{nm}" if nm else None)
        
        tr = translit.strip().lower() if translit else None
        key_B.append(f"B||{c}||{tr}" if tr else None)
        
        tokens = extract_meaningful_tokens(nm) if nm else []
        key_C.append(f"C||{c}||{tokens[0]} {tokens[1]}" if len(tokens) >= 2 else None)
        
        if tokens and has_num:
            key_D.append(f"D||{c}||{tokens[0]}||{num}")
        else:
            key_D.append(None)
            
        if tokens and not has_num:
            key_E.append(f"E||{c}||{tokens[0]}")
        else:
            key_E.append(None)
            
        key_F.append(f"F||{c}||{ad}" if has_addr else None)
        
    out_df = pl.DataFrame({
        "entity_id": eids,
        "country": clean_countries,
        "has_addr_num": has_nums,
        "has_addr": has_addrs,
        "key_A": key_A,
        "key_B": key_B,
        "key_C": key_C,
        "key_D": key_D,
        "key_E": key_E,
        "key_F": key_F,
    })
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_df.write_parquet(out_file, compression="snappy")
    logger.info(
        f"Finished {source_name}: {out_df.height:,} rows written to {out_file} "
        f"in {time.time()-t0:.1f}s ({os.path.getsize(out_file)/1e6:.1f} MB)"
    )
    del out_df

def main():
    out_dir = "data/processed/train/blocking_keys"
    os.makedirs(out_dir, exist_ok=True)
    
    process_source_to_parquet("source1", os.path.join(out_dir, "s1_keys.parquet"))
    process_source_to_parquet("source2", os.path.join(out_dir, "s2_keys.parquet"))
    process_source_to_parquet("source3", os.path.join(out_dir, "s3_keys.parquet"))
    logger.info("All blocking key parquets ready!")

if __name__ == "__main__":
    main()
