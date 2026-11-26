import unittest
import numpy as np
import pandas as pd
from src.recovery_time import (compute_k_consecutive_days, compute_k_rolling_3day_mean,
    compute_k_cumulative_mean, get_post_promo_window, find_clean_baseline)
from src.promotion_detection import phase0_flags, postprocess_promo

class RecoveryTests(unittest.TestCase):
    def test_consecutive(self):
        self.assertEqual(compute_k_consecutive_days([3, 10, 12], 10, 2), 3)
        self.assertTrue(np.isnan(compute_k_consecutive_days([10, 3, 10], 10, 2)))
    def test_moving_and_cumulative(self):
        self.assertEqual(compute_k_rolling_3day_mean([3, 9, 18], 10), 3)
        self.assertEqual(compute_k_cumulative_mean([3, 9, 18], 10), 3)
    def test_next_promotion(self):
        frame = pd.DataFrame({'is_promotion': [1, 0, 0, 1, 0]})
        self.assertEqual(len(get_post_promo_window(frame, 0)), 2)
    def test_clean_baseline(self):
        frame = pd.DataFrame({'OrderDate': pd.date_range('2024-01-01', periods=15), 'Qty': [7]*15, 'is_promotion': [0]*14+[1]})
        self.assertEqual(find_clean_baseline(frame, 14)['baseline_qty'], 7)
    def test_short_initial_window(self):
        self.assertEqual(len(phase0_flags(np.array([1.,2.,3.]),14,1.5)),3)

if __name__ == '__main__': unittest.main()
