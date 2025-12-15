import pandas as pd
import numpy as np
from offline_train_2min import train_models_2min, make_trade_outcome_label
from intraday_cache_manager import get_cache_manager

# Split 1: Train on Jan-Aug 2025, test on Sep 2025
# Split 2: Train on Jan-Sep 2025, test on Oct 2025
# Split 3: Train on Jan-Oct 2025, test on Nov 2025

splits = [
    ('2024-01-01', '2025-08-31', '2025-09-01', '2025-09-30'),
    ('2024-01-01', '2025-09-30', '2025-10-01', '2025-10-31'),
    ('2024-01-01', '2025-10-29', '2025-11-01', '2025-12-05'),
]

for train_start, train_end, test_start, test_end in splits:
    print(f"\nSplit: Train {train_start}→{train_end}, Test {test_start}→{test_end}")
    # Run training and evaluation
    # Record accuracy