#!/usr/bin/env python3
"""Live training monitor dat HTML in real-time aanpast."""

import time
import re
from pathlib import Path
from datetime import datetime

log_file = Path("training.log")
monitor_file = Path("training_monitor.html")
start_time = time.time()

def parse_log():
    """Parse trainingslogbestand."""
    metrics = {
        'epoch': '0/3',
        'batch': 0,
        'total_batches': 1391,
        'loss': '--',
        'grad_norm': '--',
        'logs': []
    }

    if not log_file.exists():
        return metrics

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[-100:]
    except:
        return metrics

    for line in lines:
        metrics['logs'].append(line.strip())

        if 'Epoch' in line and '/' in line:
            match = re.search(r'Epoch (\d+)/(\d+)', line)
            if match:
                metrics['epoch'] = f"{match.group(1)}/{match.group(2)}"

        if 'Batch' in line:
            match = re.search(r'Batch (\d+):', line)
            if match:
                metrics['batch'] = int(match.group(1))

        if 'Loss =' in line:
            match = re.search(r'Loss = ([0-9.]+)', line)
            if match:
                metrics['loss'] = match.group(1)

        if 'Grad Norm' in line:
            match = re.search(r'Grad Norm = ([0-9.]+)', line)
            if match:
                metrics['grad_norm'] = match.group(1)

    return metrics

def calculate_eta(batch, elapsed):
    """Bereken ETA."""
    if batch == 0 or elapsed < 1:
        return "Berekenen..."

    rate = batch / elapsed
    remaining = (1391 - batch) / rate if rate > 0 else 0

    h = int(remaining // 3600)
    m = int((remaining % 3600) // 60)
    s = int(remaining % 60)

    return f"{h:02d}:{m:02d}:{s:02d}"

def update_monitor():
    """Update HTML monitor."""
    metrics = parse_log()
    elapsed = time.time() - start_time

    batch_pct = min(100, (metrics['batch'] / metrics['total_batches']) * 100)
    progress = int((metrics['batch'] / (metrics['total_batches'] * 3)) * 100)
    eta = calculate_eta(metrics['batch'], elapsed)

    log_html = ""
    for log in metrics['logs'][-10:]:
        css_class = ""
        if 'Loss' in log or 'Grad' in log:
            css_class = 'loss'
        elif 'WARNING' in log or 'ERROR' in log:
            css_class = 'warning'
        log_html += f'<div class="log-entry {css_class}">{log}</div>\n'

    html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="3">
    <title>Training Monitor LIVE</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px; }}
        header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #238636; padding-bottom: 15px; }}
        h1 {{ color: #58a6ff; margin-bottom: 8px; }}
        .timer-box {{ background: #0d1117; border: 2px solid #238636; border-radius: 6px; padding: 20px; margin: 20px 0; text-align: center; }}
        .timer-display {{ font-size: 2.2em; font-weight: bold; color: #58a6ff; letter-spacing: 2px; }}
        .progress-bar {{ width: 100%; height: 24px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #238636, #3fb950); display: flex; align-items: center; justify-content: center; color: #0d1117; font-weight: bold; transition: width 0.3s; }}
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; }}
        .stat-label {{ color: #8b949e; font-size: 0.85em; margin-bottom: 5px; }}
        .stat-value {{ color: #58a6ff; font-size: 1.4em; font-weight: bold; }}
        .running {{ display: inline-block; width: 10px; height: 10px; background: #3fb950; border-radius: 50%; margin-right: 6px; animation: pulse 1.5s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .log-entry {{ background: #0d1117; border-left: 3px solid #238636; padding: 8px 12px; margin: 5px 0; font-size: 0.85em; color: #8b949e; border-radius: 3px; }}
        .log-entry.loss {{ border-left-color: #58a6ff; color: #58a6ff; }}
        .log-entry.warning {{ border-left-color: #d29922; color: #d29922; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Stap 4 Training Monitor - LIVE</h1>
            <p><span class="running"></span>Training actief...</p>
        </header>

        <div class="timer-box">
            <div class="timer-display">{eta}</div>
            <div style="color: #8b949e; margin-top: 8px;">Geschatte ETA</div>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Epoch</div>
                <div class="stat-value">{metrics['epoch']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Progress</div>
                <div class="stat-value">{progress}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Loss</div>
                <div class="stat-value">{metrics['loss']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Grad Norm</div>
                <div class="stat-value">{metrics['grad_norm']}</div>
            </div>
        </div>

        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Batch Progress</span>
                <span>{metrics['batch']} / {metrics['total_batches']}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {batch_pct}%;"></div>
            </div>
        </div>

        <div style="margin: 20px 0;">
            <div style="color: #8b949e; margin-bottom: 10px;"><strong>Recente logs:</strong></div>
            {log_html}
        </div>

        <footer style="text-align: center; margin-top: 30px; padding-top: 15px; border-top: 1px solid #30363d; color: #8b949e; font-size: 0.85em;">
            <p>Automatisch vernieuwd elke 3 seconden • {datetime.now().strftime('%H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>"""

    with open(monitor_file, 'w') as f:
        f.write(html)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"Epoch {metrics['epoch']} | Batch {metrics['batch']}/{metrics['total_batches']} | "
          f"Loss: {metrics['loss']} | ETA: {eta}")

print("🔴 Live training monitor gestart...")
print("📊 Open: training_monitor.html\n")

try:
    while True:
        update_monitor()
        time.sleep(3)
except KeyboardInterrupt:
    print("\n✅ Monitor gestopt.")
