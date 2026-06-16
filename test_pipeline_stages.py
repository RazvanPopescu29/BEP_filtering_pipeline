import unittest
import pandas as pd
import numpy as np
import hashlib
from wordfreq import zipf_frequency

from pipeline import (
    GlobalDeduplicator,
    is_char_spam,
    has_excessive_repetition,
    filter_partial_english,
    generate_minhash_array,
    CONFIG
)

class TestPipelineComprehensiveSuite(unittest.TestCase):

    def setUp(self):
        """Sets up isolated test cases and state tracking containers."""
        self.state_manager = GlobalDeduplicator()

        from pipeline import init_worker
        init_worker("lid.176.bin")
        
        self.character_spam = ". . . . . . . . . . . . . . . . . . . Low"
        self.repetition_spam = "deal only " * 10
        self.mixed_language = "Vendiendo selling fresh RDP logs"
        self.clean_english = "selling fresh credentials contact via telegram group"
        
        # Prepare the data used for testing the stateless stages
        self.raw_message = "I shouldn t be saying this...but all the same I m so so grateful to you sir I pray God bless your hard works"
        self.evasive_duplicate = "    I shouldn t be saying this...but all the same I m so so grateful to you sir I pray God bless your hard works"
        
        self.lsh_original = (
            "ALERT: Fresh remote desktop access credentials available for corporate infrastructure network fields. "
            "Guaranteed administrative access keys are verified, stable, and fully tested on consumer grade hardware systems. "
            "All logs include clean system configurations, clear baseline metrics, database schemas, and compromised server network routes. "
            "This package is ready for immediate deployment in active data engineering pipelines. "
            "Click on the link and see what am talking about."
        )
        self.lsh_mutation = (
            "ALERT: Fresh remote desktop access credentials available for corporate infrastructure network fields. "
            "Guaranteed administrative access keys are verified, stable, and fully tested on consumer grade hardware systems. "
            "All logs include clean system configurations, clear baseline metrics, database schemas, and compromised server network routes. "
            "This package is ready for immediate deployment in active data engineering pipelines. "
            "Click on the link and see what I have for you."
        )

    # Test stateless stages
    def test_stateless_heuristics(self):
        """Validates character spam, token repetition spam, and Zipf log-frequency language surgery."""
        # Test character spam
        self.assertTrue(is_char_spam(self.character_spam))
        self.assertFalse(is_char_spam(self.clean_english))
        
        # Test repetition
        self.assertTrue(has_excessive_repetition(self.repetition_spam.strip()))
        self.assertFalse(has_excessive_repetition(self.clean_english))
        
        # Test non-English removal
        processed_text = filter_partial_english(self.mixed_language)
        self.assertIsNotNone(processed_text)
        self.assertNotIn("Vendiendo", processed_text)
        self.assertIn("selling", processed_text)
        self.assertIn("fresh", processed_text)

        # Micro-messages
        self.assertIsNone(generate_minhash_array("Es tuya"))

    # Test stateful stages
    def test_stateful_deduplication_and_lsh_tracking(self):
        """Validates exact Bloom filters, clean Bloom filters, and near-duplicate LSH tracking."""
        
        df_bloom = pd.DataFrame([{"text": self.raw_message}, {"text": self.evasive_duplicate}])
        df_bloom["raw_md5"] = df_bloom["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())
        
        df_bloom, raw_deleted = self.state_manager.apply_bloom(df_bloom, "raw_md5")
        self.assertEqual(len(df_bloom), 2)
        
        df_bloom["text"] = df_bloom["text"].str.replace(r"\s+", " ", regex=True).str.strip()
        df_bloom["clean_md5"] = df_bloom["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())
        
        df_bloom, clean_deleted = self.state_manager.apply_bloom(df_bloom, "clean_md5")
        self.assertEqual(len(df_bloom), 1)
        self.assertEqual(len(clean_deleted), 1)

        lsh_state_manager = GlobalDeduplicator()

        df_lsh = pd.DataFrame([
            {"text": self.lsh_original, "raw_md5": "id_orig"},
            {"text": self.lsh_mutation, "raw_md5": "id_mutated"}
        ])
        df_lsh["minhash_values"] = df_lsh["text"].apply(generate_minhash_array)
        
        kept_orig, deleted_orig = lsh_state_manager.apply_lsh(df_lsh.iloc[[0]].copy())
        self.assertEqual(len(kept_orig), 1)
        self.assertEqual(len(deleted_orig), 0)
        
        kept_mut, deleted_mut = lsh_state_manager.apply_lsh(df_lsh.iloc[[1]].copy())
        self.assertEqual(len(kept_mut), 0)
        self.assertEqual(len(deleted_mut), 1)
        
        matched_original = deleted_mut["matched_with_original_md5"].iloc[0]
        change_location = deleted_mut["change_location"].iloc[0]
        
        self.assertEqual(matched_original, "id_orig")
        self.assertEqual(change_location, "End")

    def test_structural_metadata_and_placeholder_gates(self):
        """Validates structural pipeline stages requiring DataFrame schema evaluations (Bots and Placeholders)."""
        from pipeline import process_chunk_stateless
        
        mock_chunk = pd.DataFrame([
            {"sender_type": "bot", "text": "Automated alert: new user joined the chat."},
            {"sender_type": "user", "text": "content could not be displayed"},
            {"sender_type": "user", "text": "ALERT: Fresh remote desktop access credentials available."}
        ])
        
        _, deleted_telemetry = process_chunk_stateless(mock_chunk)
        
        self.assertIn("deleted_bots.csv", deleted_telemetry)
        self.assertEqual(len(deleted_telemetry["deleted_bots.csv"]), 1)
        self.assertEqual(deleted_telemetry["deleted_bots.csv"]["text"].iloc[0], "Automated alert: new user joined the chat.")
        
        self.assertIn("deleted_placeholders.csv", deleted_telemetry)
        self.assertEqual(len(deleted_telemetry["deleted_placeholders.csv"]), 1)
        self.assertEqual(deleted_telemetry["deleted_placeholders.csv"]["text"].iloc[0], "content could not be displayed")

if __name__ == "__main__":
    unittest.main()