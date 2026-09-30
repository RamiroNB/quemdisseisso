"""Local preview of the portal with browser caching off, so an edit never mixes with stale CSS/JS.

Usage: python3 serve.py [port]      (default 8803; GitHub Pages needs none of this)
"""
import http.server
import os
import socket
import sys
import threading


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


class Server6(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8803
    print(f"http://localhost:{port}", flush=True)
    # loopback only, on IPv4 AND IPv6: here "localhost" resolves to ::1 first, which is what VS Code's port
    # forwarding connects to; a 127.0.0.1-only server looks down through the forward
    try:
        s6 = Server6(("::1", port), NoCache)
        threading.Thread(target=s6.serve_forever, daemon=True).start()
    except OSError as e:
        print(f"no IPv6 loopback ({e}); serving IPv4 only", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), NoCache).serve_forever()
