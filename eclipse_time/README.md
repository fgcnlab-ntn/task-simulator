# 6/27 Eclipse-time experiment

This experiment measures how long satellites stay in eclipse in the current
Starlink-like Walker shell configuration.

## Method

The experiment runs the starlink for `43200s` with task execution
disabled. For each satellite, it tracks complete transitions:

```text
sunlit -> eclipse -> sunlit
```

Each complete eclipse interval contributes one duration:

```text
duration_s = end_s - start_s
```

## Result

From `11399` complete eclipse intervals:

```text
low   = 1560 s  (26.00 min)
mean  = 1915 s  (31.92 min)
high  = 2160 s  (36.00 min)
```

## Files

- `eclipse_intervals.jsonl` — one complete eclipse interval per line
- `eclipse_samples.jsonl` — per-step sunlit/eclipse satellite counts
- `eclipse_time_summary.json` — numeric summary
- `eclipse_time_bar.svg` — boxplot summary
- `eclipse_time_pdf.svg` — interval-duration distribution
- `eclipse_time_cdf.svg` — cumulative distribution
- `run_config.json` — effective experiment configuration

Regenerate with:

```bash
python3 tools/eclipse_time_experiment.py --out eclipse_time
```
