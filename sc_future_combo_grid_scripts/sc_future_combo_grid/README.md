# SC future-combination grid

Copy these files into:

    ~/KV-Aware-vLLM/local_repro/grid_search/

Standard 250-question run:

    python local_repro/grid_search/run_future_combo_grid.py \
      --max-questions 250

Print the exact configurations without submitting:

    python local_repro/grid_search/run_future_combo_grid.py \
      --max-questions 250 \
      --print-grid

Run the same grid with LMCache disk data on a dedicated scratch directory:

    python local_repro/grid_search/run_future_combo_grid.py \
      --max-questions 250 \
      --lmcache-data-dir /scratch/$USER/lmcache_future_grid

The five runs are sequential. When --lmcache-data-dir is given, every job
reuses that exact directory; run_driver.sh deletes/recreates it at job start,
preventing scratch usage from accumulating across the grid.

Important: dedicate that scratch directory to this grid while it is running.

Experiment 4 note:
The meeting slide's experiment 1 and experiment 4 are identical under the
current defaults (batch=50, worker gate disabled, disk-put gate disabled).
This runner makes experiment 4 useful by enabling worker admission max=4 in
addition to scheduler max=4 and max_num_seqs=12.
