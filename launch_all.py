"""Launcher script to run all 9 versions (myapp0 -> myapp13) simultaneously on distinct local ports."""
import subprocess
import sys
import time

APPS = [
    ("myapp0.py", 8500),
    ("myapp1.py", 8501),
    ("myapp2.py", 8502),
    ("myapp5.py", 8505),
    ("myapp7.py", 8507),
    ("myapp8.py", 8508),
    ("myapp9.py", 8509),
    ("myapp10.py", 8510),
    ("myapp13.py", 8513),
]


def main():
    procs = []
    for app, port in APPS:
        print(f"Starting {app} at http://localhost:{port}", flush=True)
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app,
                "--server.port",
                str(port),
                "--server.headless",
                "true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        procs.append((app, port, p))

    print("\nAll 9 apps are running! Press Ctrl+C to stop all.", flush=True)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping all apps...", flush=True)
        for _, _, p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
