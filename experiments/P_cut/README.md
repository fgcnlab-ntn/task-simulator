# P_cut

This experiment estimates the eclipse-side energy consumption of one satellite
when its CPU is fully active for the whole eclipse interval.

## Method

Fixed assumptions:

- one satellite
- eclipse duration: 32 minutes = 1920 s
- idle power: 4 W
- battery capacity: 216,000 J
- initial battery: 100%
- CPU power sweep: 0–60 W

Energy model:

```text
total energy = (CPU power + idle power) × eclipse duration
```

The plots compare this energy against the usable battery energy before reaching
minimum safe battery limits of 20%, 50%, and 70%.

## Results

| Minimum safe battery | Usable energy | CPU power limit |
| --- | ---: | ---: |
| 20% | 172.8 kJ | 86.0 W |
| 50% | 108.0 kJ | 52.25 W |
| 70% | 64.8 kJ | 29.75 W |

With a 70% minimum safe battery limit, a CPU power near 30 W reaches the safe
energy boundary.

## Files

- `p_cut_results.csv`: power sweep data
- `p_cut_energy_safe_20pct.svg`: plot for 20% minimum safe battery
- `p_cut_energy_safe_50pct.svg`: plot for 50% minimum safe battery
- `p_cut_energy_safe_70pct.svg`: plot for 70% minimum safe battery
