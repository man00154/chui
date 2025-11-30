import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def simulate_timeseries(chillers, days=7):
    data = []
    for ch in chillers:
        t = datetime.now() - timedelta(days=days)
        for _ in range(days * 96):  # 15-minute intervals
            ambient = np.random.uniform(28, 45)
            it_load = np.random.uniform(40, 95)
            chw_in = np.random.uniform(5, 14)

            base = 260 + (ambient - 30) * 10 + (it_load - 40) * 3 + (chw_in - 6) * 8
            predicted = base * np.random.uniform(0.92, 1.08)
            actual = base * np.random.uniform(0.85, 1.25)
            anomaly = (actual - predicted) / max(predicted, 1)

            data.append({
                "time": t,
                "chiller": ch,
                "ambient": ambient,
                "it_load": it_load,
                "chw_in": chw_in,
                "power_predicted": predicted,
                "power_actual": actual,
                "anomaly_score": anomaly
            })
            t += timedelta(minutes=15)
    return pd.DataFrame(data)

def simulate_maintenance(chillers):
    return pd.DataFrame({
        "chiller": chillers,
        "due_days": [random.randint(5, 180) for _ in chillers],
        "priority": [random.choice(["Low","Medium","High"]) for _ in chillers]
    })

def simulate_anomaly_summary(chillers):
    return pd.DataFrame({
        "chiller": chillers,
        "count": [random.randint(5, 45) for _ in chillers]
    })
