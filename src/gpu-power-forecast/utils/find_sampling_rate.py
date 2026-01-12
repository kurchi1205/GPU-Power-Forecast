import json
import numpy as np
from datetime import datetime

json_path = "../../metrics_log/merged_log.json"   # change if needed

# ---- load timestamps ----
times = []
with open(json_path) as f:
    for line in f:
        try:
            rec = json.loads(line)
            times.append(datetime.fromisoformat(rec["timestamp"]))
        except Exception:
            continue

times = np.array(times)

# ---- convert to seconds ----
time_s = np.array([(t - times[0]).total_seconds() for t in times])

# ---- time deltas ----
dt = np.diff(time_s)

median_dt = np.median(dt)
mean_dt   = np.mean(dt)

sampling_rate_median = 1.0 / median_dt
sampling_rate_mean   = 1.0 / mean_dt

print(f"Samples loaded: {len(times)}")
print(f"Median dt: {median_dt:.6f} s → {sampling_rate_median:.2f} Hz")
print(f"Mean dt:   {mean_dt:.6f} s → {sampling_rate_mean:.2f} Hz")
