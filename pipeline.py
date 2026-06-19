import csv
import difflib
from functools import lru_cache
import gc
import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from typing import Dict, List, Tuple, Any, Optional, Iterator

from datasketch import MinHash, MinHashLSH
import fasttext
import numpy as np
import pandas as pd
import psutil
from pybloom_live import BloomFilter
from wordfreq import zipf_frequency

# Configuration configurations with environment variable overrides
CONFIG: Dict[str, Any] = {
    "batch_size": int(os.environ.get("PIPELINE_BATCH_SIZE", 100_000)),
    "fasttext_min_conf": float(os.environ.get("PIPELINE_FT_CONF", 0.6)),
    "min_english_ratio": float(os.environ.get("PIPELINE_MIN_ENG_RATIO", 0.6)),
    "min_words": int(os.environ.get("PIPELINE_MIN_WORDS", 5)),
    "lsh_threshold": float(os.environ.get("PIPELINE_LSH_THRESH", 0.85)),
    "num_perm": int(os.environ.get("PIPELINE_NUM_PERM", 128)),
    "bloom_capacity": int(os.environ.get("PIPELINE_BLOOM_CAP", 50_000_000)),
    "bloom_error": float(os.environ.get("PIPELINE_BLOOM_ERR", 0.001)),
    "min_unique_word_ratio": float(os.environ.get("PIPELINE_MIN_UNIQUE_RATIO", 0.25)),  
    "min_alpha_ratio": float(os.environ.get("PIPELINE_MIN_ALPHA_RATIO", 0.4)),  
    "max_worker_cores": int(os.environ.get("PIPELINE_MAX_WORKERS", 6))       
}

def resource_monitor(stop_event: threading.Event, log_file: str = "hardware_usage.csv") -> None:
    """
    Background thread to log CPU and RAM utilization.
    """
    print("Started Hardware Monitor")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Seconds", "CPU_Percent", "RAM_GB"])
        
        start_time: float = time.time()
        while not stop_event.is_set():
            current_time: float = round(time.time() - start_time, 1)
            cpu: float = psutil.cpu_percent(interval=1)
            ram: float = psutil.virtual_memory().used / (1024 ** 3) 
            
            writer.writerow([current_time, cpu, round(ram, 2)])

# Global model variable for the parallel workers
global_ft_model: Optional[fasttext.FastText._FastText] = None

def init_worker(fasttext_model_path: str) -> None:
    """
    Loads the FastText language model once for a new process.
    """
    global global_ft_model
    global_ft_model = fasttext.load_model(fasttext_model_path)

@lru_cache(maxsize=500_000)
def is_english_word(word: str) -> bool:
    """
    Checks if a word is common in the English language using Zipf frequency.
    """
    return zipf_frequency(word, "en") > 3.0

def filter_partial_english(text: str) -> Optional[str]:
    """
    Filters text based on the proportion of valid English words it contains.
    """
    words: List[str] = str(text).split()
    if len(words) < CONFIG["min_words"]: 
        return None
    english_words: List[str] = [w for w in words if is_english_word(w)]
    if not words: 
        return None
    ratio: float = len(english_words) / len(words)
    if ratio < CONFIG["min_english_ratio"]: 
        return None
    if ratio > 0.9: 
        return text
    return " ".join(english_words)

def generate_minhash_array(text: str) -> Optional[np.ndarray]:
    """
    Breaks text into 3-word overlapping chunks (shingles) to build a MinHash signature.
    """
    words: List[str] = str(text).split()
    if len(words) < 3: 
        return None
    m: MinHash = MinHash(num_perm=CONFIG["num_perm"])
    for i in range(len(words) - 2):
        m.update(f"{words[i]} {words[i+1]} {words[i+2]}".encode("utf-8"))
    return m.hashvalues

def is_char_spam(text: str) -> bool:
    """
    Checks if a message consists mostly of non-alphabetical characters and not letters or numbers.
    """
    if not text: 
        return True
        
    text_no_spaces: str = str(text).replace(" ", "")
    if len(text_no_spaces) == 0: 
        return True
        
    valid_chars: int = sum(c.isalnum() for c in text_no_spaces)
    return (valid_chars / len(text_no_spaces)) < CONFIG["min_alpha_ratio"]

def has_excessive_repetition(text: str) -> bool:
    """
    Identifies spam by checking if a message only repeats the same few words.
    """
    words: List[str] = text.split()
    if not words:
        return True
    return (len(set(words)) / len(words)) < CONFIG["min_unique_word_ratio"]

def process_chunk_stateless(chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs stateless cleaning filters (regex, length limits, language checking) on a data chunk.
    """
    deleted_logs: Dict[str, pd.DataFrame] = {
        "deleted_bots.csv": pd.DataFrame(),
        "deleted_placeholders.csv": pd.DataFrame(),
        "deleted_empty_after_norm.csv": pd.DataFrame(),
        "deleted_micro_spam.csv": pd.DataFrame(),
        "deleted_mega_dumps.csv": pd.DataFrame(),
        "deleted_char_spam.csv": pd.DataFrame(),
        "deleted_non_english.csv": pd.DataFrame(),
        "deleted_repetition_spam.csv": pd.DataFrame()
    }

    # Removes empty chunks of text.
    if chunk.empty or "text" not in chunk.columns:
        return pd.DataFrame(), deleted_logs

    # Removes bots
    if "sender_type" in chunk.columns:
        bot_mask: pd.Series = chunk["sender_type"] == "bot"
        deleted_logs["deleted_bots.csv"] = chunk[bot_mask].copy()
        chunk = chunk[~bot_mask]

    chunk = chunk.dropna(subset=["text"])
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Removes system placeholders
    placeholder_mask: pd.Series = chunk["text"].str.contains(r"^content could not be displayed$", case=False, na=False)
    deleted_logs["deleted_placeholders.csv"] = chunk[placeholder_mask].copy()
    chunk = chunk[~placeholder_mask]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Hashes messages before normalization to capture identical messages
    chunk["raw_md5"] = chunk["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())

    chunk["text"] = chunk["text"].astype(str)
    # Remove URL's
    chunk["text"] = chunk["text"].str.replace(r"http\S+|www\S+", " ", regex=True)
    # Removes emojis
    chunk["text"] = chunk["text"].str.replace(r"[\U00010000-\U0010ffff]", "", regex=True) 

    # Replaces leetspeak $ with normal s
    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])\$|\$(?=[a-zA-Z])", "s", regex=True)
    # Replaces leetspeak @ with normal a
    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])@|@(?=[a-zA-Z])", "a", regex=True)
    # Replaces leetspeak 0 with normal o
    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])0|0(?=[a-zA-Z])", "o", regex=True)
    # Replaces leetspeak 3 with normal e
    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])3|3(?=[a-zA-Z])", "e", regex=True)

    # Keeps only alphanumeric characters and a few accepted ones
    chunk["text"] = chunk["text"].str.replace(r"[^a-zA-Z0-9\s$€£.,]", " ", regex=True)
    # Removes extra whitespaces
    chunk["text"] = chunk["text"].str.replace(r"\s+", " ", regex=True).str.strip() 

    # Removes empty chunks of text.
    empty_mask: pd.Series = chunk["text"] == ""
    deleted_logs["deleted_empty_after_norm.csv"] = chunk[empty_mask].copy()
    chunk = chunk[~empty_mask]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Apply text length constraints (keeps only messages between 30 and 2500 characters)
    too_short: pd.Series = chunk["text"].str.len() < 30
    deleted_logs["deleted_micro_spam.csv"] = chunk[too_short].copy()
    chunk = chunk[~too_short]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    too_long: pd.Series = chunk["text"].str.len() > 2500
    deleted_logs["deleted_mega_dumps.csv"] = chunk[too_long].copy()
    chunk = chunk[~too_long]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Removes character spam messages
    char_spam_mask: pd.Series = chunk["text"].apply(is_char_spam)
    deleted_logs["deleted_char_spam.csv"] = chunk[char_spam_mask].copy()
    chunk = chunk[~char_spam_mask]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Hashes clean text
    chunk["clean_md5"] = chunk["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())

    # Evaluates language with FastText
    texts: List[str] = chunk["text"].tolist()
    labels, probs = global_ft_model.predict(texts)
    chunk["lang"] = [l[0] for l in labels]
    chunk["conf"] = [p[0] for p in probs]

    ft_mask: pd.Series = (chunk["lang"] == "__label__en") & (chunk["conf"] >= CONFIG["fasttext_min_conf"])
    deleted_logs["deleted_non_english.csv"] = chunk[~ft_mask].drop(columns=["lang", "conf"]).copy()
    chunk = chunk[ft_mask].drop(columns=["lang", "conf"])
    if chunk.empty: return pd.DataFrame(), deleted_logs

    chunk["text"] = chunk["text"].apply(filter_partial_english)
    chunk = chunk.dropna(subset=["text"])
    if chunk.empty: return pd.DataFrame(), deleted_logs

    # Drops messages with high repetition of words
    repetition_mask: pd.Series = chunk["text"].apply(has_excessive_repetition)
    deleted_logs["deleted_repetition_spam.csv"] = chunk[repetition_mask].copy()
    chunk = chunk[~repetition_mask]
    if chunk.empty: return pd.DataFrame(), deleted_logs

    chunk["minhash_values"] = chunk["text"].apply(generate_minhash_array)

    return chunk, deleted_logs

def log_deleted_rows(df: pd.DataFrame, path: str) -> None:
    """
    Appends filtered text records into separate CSV files.
    """
    if df is None or df.empty: 
        return
    mode: str = "a" if os.path.exists(path) else "w"
    df.to_csv(path, index=False, mode=mode, header=(mode == "w"))

class GlobalDeduplicator:
    """
    Stateful memory manager handling global Bloom Filters and MinHash LSH indexing.
    """
    def __init__(self) -> None:
        self.seen_raw: BloomFilter = BloomFilter(CONFIG["bloom_capacity"], CONFIG["bloom_error"])
        self.seen_clean: BloomFilter = BloomFilter(CONFIG["bloom_capacity"], CONFIG["bloom_error"])
        self.lsh: MinHashLSH = MinHashLSH(threshold=CONFIG["lsh_threshold"], num_perm=CONFIG["num_perm"])
        self.counter: int = 0
        self.md5_to_text: Dict[str, str] = {} 

    def apply_bloom(self, chunk: pd.DataFrame, md5_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Finds exact duplicates by looking up MD5 string hashes in a Bloom Filter.
        """
        if chunk.empty: return chunk, pd.DataFrame()
        bloom_mask: pd.Series = chunk[md5_col].apply(lambda h: h in self.seen_raw if md5_col == 'raw_md5' else h in self.seen_clean)
        local_dupe_mask: pd.Series = chunk.duplicated(subset=[md5_col], keep='first')
        final_mask: pd.Series = bloom_mask | local_dupe_mask
        deleted: pd.DataFrame = chunk[final_mask].copy()
        chunk = chunk[~final_mask]
        
        # Commit newly discovered unique hashes into the active filter state
        for h in chunk[md5_col]:
            if md5_col == 'raw_md5': 
                self.seen_raw.add(h)
            else: 
                self.seen_clean.add(h)
        return chunk, deleted

    def determine_change_location(self, orig_text: str, curr_text: str) -> str:
        """
        Calculates where a near-duplicate text was changed (Beginning, Middle, or End).
        """
        if not orig_text or not curr_text: 
            return "Unknown"
            
        orig_words: List[str] = str(orig_text).split()
        curr_words: List[str] = str(curr_text).split()
        if len(orig_words) == 0:
            return "Unknown"

        matcher: difflib.SequenceMatcher = difflib.SequenceMatcher(None, orig_words, curr_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                position_ratio: float = i1 / len(orig_words)
                if position_ratio < 0.33:
                    return "Beginning"
                elif position_ratio > 0.66:
                    return "End"
                else:
                    return "Middle"
                    
        return "Scattered Minor Edits"
    
    def apply_lsh(self, chunk) -> Tuple[pd.DataFrame, pd.DataFrame]:
        keep_mask = []
        to_insert = []
        matched_original_ids = []
        change_locations = []
        
        for hashvals, current_msg_id, current_text in zip(chunk["minhash_values"], chunk["raw_md5"], chunk["text"]):
            if hashvals is None or not isinstance(hashvals, np.ndarray):
                keep_mask.append(False)
                matched_original_ids.append("Invalid/Short")
                change_locations.append("N/A")
                continue
            
            m = MinHash(num_perm=CONFIG["num_perm"], hashvalues=hashvals)
            matches = self.lsh.query(m)
            
            if matches:
                matched_id = matches[0]
                keep_mask.append(False)
                matched_original_ids.append(matched_id)
                
                orig_text = self.md5_to_text.get(matched_id, "")
                location = self.determine_change_location(orig_text, current_text)
                change_locations.append(location)
            else:
                keep_mask.append(True)
                matched_original_ids.append(None)
                change_locations.append(None)
                
                to_insert.append((current_msg_id, m))
                self.md5_to_text[current_msg_id] = current_text
                self.counter += 1
                
        if to_insert:
            with self.lsh.insertion_session() as session:
                for key, m in to_insert:
                    session.insert(key, m)
                    
        mask_series = pd.Series(keep_mask, index=chunk.index)
        
        deleted_chunk = chunk[~mask_series].copy()
        deleted_chunk["matched_with_original_md5"] = [m_id for msk, m_id in zip(keep_mask, matched_original_ids) if not msk]
        deleted_chunk["change_location"] = [loc for msk, loc in zip(keep_mask, change_locations) if not msk]
        
        return chunk[mask_series], deleted_chunk

def process_dataset_parallel_unordered(input_file: str, output_file: str, fasttext_model_path: str) -> None:
    """
    Coordinates data streaming and maps data chunks to asynchronous process pool workers.
    """
    start: float = time.time()
    print("Initializing Unordered Parallel Pipeline")

    # Clean up previous operation outputs and file logs
    log_files: List[str] = [
        "deleted_bots.csv", "deleted_placeholders.csv", "deleted_raw_dupes.csv",
        "deleted_clean_dupes.csv", "deleted_empty_after_norm.csv",
        "deleted_non_english.csv", "deleted_near_dupes.csv",
        "deleted_char_spam.csv", "deleted_repetition_spam.csv",
        "deleted_micro_spam.csv", "deleted_mega_dumps.csv", output_file
    ]
    for f in log_files:
        if os.path.exists(f): 
            os.remove(f)

    state_manager: GlobalDeduplicator = GlobalDeduplicator()
    reader = pd.read_csv(
        input_file, chunksize=CONFIG["batch_size"], dtype=str, usecols=["text", "sender_type"]
    )
    reader_iter: Iterator[pd.DataFrame] = iter(reader)
    
    num_workers: int = min(CONFIG["max_worker_cores"], max(1, multiprocessing.cpu_count() - 1))
    print(f"Firing up {num_workers} CPU cores for UNORDERED processing")

    # Launchs multiprocessing pool
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(fasttext_model_path,)) as executor:
        active_futures: Dict[Future, int] = {}
        batch_counter: int = 0

        # Feeds initial chunks into the process workers queue
        for _ in range(num_workers * 2):
            try:
                chunk: pd.DataFrame = next(reader_iter)
                batch_counter += 1
                future: Future = executor.submit(process_chunk_stateless, chunk)
                active_futures[future] = batch_counter
            except StopIteration:
                break

        # Proccesses completed batches out of order when they finish
        first_save: bool = True
        while active_futures:
            done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                batch_num: int = active_futures.pop(future)
                processed_chunk, deleted_logs = future.result()

                print(f"\n Merging Batch {batch_num} into Main Process ")

                # Logs independent stateless deletions returned by the worker
                for log_name, df_del in deleted_logs.items():
                    log_deleted_rows(df_del, log_name)

                # Does stateful main process checks (Exact Hashing and Fuzzy LSH)
                if not processed_chunk.empty:
                    processed_chunk, raw_dupes = state_manager.apply_bloom(processed_chunk, "raw_md5")
                    log_deleted_rows(raw_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_raw_dupes.csv")

                    processed_chunk, clean_dupes = state_manager.apply_bloom(processed_chunk, "clean_md5")
                    log_deleted_rows(clean_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_clean_dupes.csv")

                    processed_chunk, near_dupes = state_manager.apply_lsh(processed_chunk)
                    log_deleted_rows(near_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_near_dupes.csv")

                    processed_chunk = processed_chunk.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore')

                # Appends unique rows to final filtered dataset
                mode, header = ("w", True) if first_save else ("a", False)
                processed_chunk.to_csv(output_file, index=False, mode=mode, header=header)
                first_save = False

                print(f"       Batch {batch_num} Saved | Kept {len(processed_chunk):,} unique rows")
                gc.collect()  # Removes old data chunk references from RAM

                # Queues up next data chunk immediately to keep workers working
                try:
                    next_chunk: pd.DataFrame = next(reader_iter)
                    batch_counter += 1
                    new_future: Future = executor.submit(process_chunk_stateless, next_chunk)
                    active_futures[new_future] = batch_counter
                except StopIteration:
                    pass

    print(f"\nUnordered Pipeline Complete in {int((time.time() - start)//60)}m {int((time.time() - start)%60)}s")
    print(f"Final LSH Index size: {state_manager.counter:,} messages")

if __name__ == "__main__":
    stop_monitor: threading.Event = threading.Event()
    
    # Run hardware logging script inside an isolated thread
    monitor_thread: threading.Thread = threading.Thread(target=resource_monitor, args=(stop_monitor,))
    monitor_thread.start()
    
    # Starts the pipeline
    process_dataset_parallel_unordered("unfiltered-messages.csv", "filtered_dataset.csv", "lid.176.bin")
    
    # Shutdown resource tracking
    stop_monitor.set()
    monitor_thread.join()
    print("Hardware log saved to hardware_usage.csv")
