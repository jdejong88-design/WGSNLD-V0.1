#!/usr/bin/env python3
"""GPU temperatuur monitor — alert als > 90°C"""

import subprocess
import time
import sys

def get_gpu_temp():
    """Lees GPU temperatuur via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        temp = int(result.stdout.strip())
        return temp
    except Exception as e:
        print(f"Error reading temp: {e}")
        return None

print("[*] GPU Temperatuur Monitor")
print("-" * 50)

alert_threshold = 90
warning_threshold = 85
check_interval = 60  # seconden

consecutive_alerts = 0

while True:
    try:
        temp = get_gpu_temp()
        if temp is None:
            continue

        status = "✓ OK"
        if temp > alert_threshold:
            status = "⚠️  ALERT"
            consecutive_alerts += 1
            if consecutive_alerts >= 3:
                print(f"\n[CRITICAL] GPU {temp}°C gedurende 3+ checks!")
                print("[ACTION] Voer uit: python 04_train_model.py --stop")
                sys.exit(1)
        elif temp > warning_threshold:
            status = "⚠️  WARM"
            consecutive_alerts = 0
        else:
            consecutive_alerts = 0

        print(f"[{time.strftime('%H:%M:%S')}] GPU Temp: {temp}°C {status}")
        time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n[OK] Monitor gestopt")
        break
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(check_interval)
