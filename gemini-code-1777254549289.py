import json
import time
import os
import re

# Paden (zorg dat deze kloppen met je mappenstructuur)
LOG_FILE = "training.log"
STATS_FILE = "stats.json"

def parse_log():
    if not os.path.exists(LOG_FILE):
        print(f"[-] Fout: {LOG_FILE} niet gevonden!")
        return
    
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[-] Fout bij lezen log: {e}")
        return

    if not lines:
        print("[!] Logbestand is nog leeg...")
        return

    history = []
    current_stats = {}
    
    # Verbeterde Regex die exact past op jouw log-formaat
    # Voorbeeld: [BATCH 001/1391] Loss: 9.2134 | Grad: 0.85
    regex = r"BATCH\s+(\d+)/(\d+).*?Loss:\s+([\d\.]+).*?Grad:\s+([\d\.]+)"
    
    for line in lines:
        match = re.search(regex, line)
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
        output = {
            "current": current_stats,
            "history": history[-100:], # Laatste 100 voor de grafiek
            "last_update": time.strftime("%H:%M:%S")
        }
        with open(STATS_FILE, "w") as f:
            json.dump(output, f)
        print(f"[+] Update: Batch {current_stats['batch']} verwerkt. stats.json is bijgewerkt.")
    else:
        print("[?] Geen geldige trainingsdata gevonden in de laatste regels van de log.")

if __name__ == "__main__":
    print("=== WatergeusLLM Data Bridge Debug Mode ===")
    while True:
        parse_log()
        time.sleep(2) # Iets langer wachten voor stabiliteit