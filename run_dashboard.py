"""
Quick Launcher for Crop Yield Prediction Interactive Dashboard
Launches a local HTTP server and opens the dashboard in the default browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    os.chdir(DIRECTORY)
    index_file = os.path.join(DIRECTORY, "index.html")
    if not os.path.exists(index_file):
        print("Error: index.html not found! Run 'python crop_yield_pipeline.py' first.")
        sys.exit(1)

    url = f"http://localhost:{PORT}/index.html"
    print("=" * 60)
    print("  Harvest & Heat: Crop Yield Under a Changing Climate")
    print("  Interactive ML Web Dashboard")
    print("=" * 60)
    print(f"Serving at: {url}")
    print("Opening your browser now... Press Ctrl+C to stop the server.")

    # Try binding to port
    for p in range(PORT, PORT + 20):
        try:
            with socketserver.TCPServer(("", p), Handler) as httpd:
                webbrowser.open(f"http://localhost:{p}/index.html")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nServer stopped.")
                    break
            break
        except OSError:
            continue

if __name__ == "__main__":
    main()
