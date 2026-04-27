import json
import time
import os
import re

LOG_FILE = "training.log"
STATS_FILE = "stats.json"

def parse_log():
    if not os.path.exists(LOG_FILE):
        return
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        if not lines: return

    # Pak de laatste 50 regels voor de grafiek
    history = []
    current_stats = {}
    
    for line in lines:
        # Match formaat: [BATCH 001/1391] Loss: 9.8765 | Grad: 1.23
        match = re.search(r"BATCH (\d+)/(\d+).+Loss: ([\d\.]+).+Grad: ([\d\.]+)", line)
        if match:
            batch, total, loss, grad = match.groups()
            data_point = {
                "batch": int(batch),
                "loss": float(loss),
                "grad": float(grad)
            }
            history.append(data_point)
            current_stats = data_point
            current_stats["total_batches"] = int(total)

    if current_stats:
        # Bewaar alleen de laatste 100 punten voor de grafiek-performance
        output = {
            "current": current_stats,
            "history": history[-100:]
        }
        with open(STATS_FILE, "w") as f:
            json.dump(output, f)

if __name__ == "__main__":
    print("Mission Control Data Bridge gestart...")
    while True:
        parse_log()
        time.sleep(1) # Update elke seconde