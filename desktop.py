"""
╔══════════════════════════════════════════════════════════════════╗
║  FORAX Desktop Application — Entry Point                        ║
║  Forensic Analysis & eXtraction Platform                        ║
║  Lahore Garrison University — BSDFCS 2025                       ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  This is the main entry point for running FORAX as a desktop app.
  It launches a native OS window (via pywebview) with no browser 
  chrome — no URL bar, no refresh button, no back/forward.
  
  This prevents investigators from accidentally refreshing 
  during evidence extraction, which would break the SSE 
  (Server-Sent Events) connection and lose the live feed.

ARCHITECTURE:
  ┌─────────────────────────────────────────────┐
  │  Native Desktop Window (pywebview)           │
  │  ┌─────────────────────────────────────────┐ │
  │  │  Internal Flask Server (localhost)       │ │
  │  │  - Serves HTML templates                │ │
  │  │  - Handles API requests                 │ │
  │  │  - Manages SSE extraction streams       │ │
  │  └─────────────────────────────────────────┘ │
  │  Background threads:                         │
  │  - ADB extraction workers                    │
  │  - AI analysis (CNN + NLP)                   │
  │  - Email notifications                       │
  └─────────────────────────────────────────────┘

HOW IT WORKS:
  1. Finds a free port on localhost (avoids conflicts)
  2. Starts Flask server in a background daemon thread
  3. Waits until Flask is ready (max 15 seconds)
  4. Opens a pywebview native window pointing to Flask
  5. Tries GUI backends: CEF → EdgeChromium → MSHTML
  6. When user closes window, the app exits cleanly

USAGE:
  python desktop.py          (launch desktop app)
  python app.py              (fallback: run as web app)

DEPENDENCIES:
  pip install pywebview[cef]  (recommended for Windows)
  pip install flask bcrypt reportlab
"""

import threading   # For running Flask in a background thread
import time        # For delays and timeout polling
import os          # For file/directory operations
import sys         # For sys.exit() on errors
import socket      # For finding free ports and server readiness checks


def find_free_port():
    """
    Find a random free port on localhost.
    
    We bind to port 0 which tells the OS to assign any available port.
    This avoids conflicts with other services (e.g., another Flask instance).
    
    Returns:
        int: A free port number (e.g., 54321)
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))        # Bind to any free port
        return s.getsockname()[1]         # Return the assigned port number


def wait_for_server(port, timeout=15):
    """
    Wait until the Flask server is accepting connections.
    
    Polls the server every 200ms until it responds or times out.
    This is needed because Flask starts in a background thread and 
    takes a moment to initialize. We must wait before opening the window.
    
    Args:
        port (int): The port Flask is running on
        timeout (int): Maximum seconds to wait (default: 15)
    
    Returns:
        bool: True if server is ready, False if timed out
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            # Try to connect to the Flask server
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return True  # Connection successful — server is ready
        except (ConnectionRefusedError, OSError):
            time.sleep(0.2)  # Server not ready yet — wait and retry
    return False  # Timed out — server failed to start


def start_flask(port):
    """
    Start the Flask server in the current thread.
    
    This function is called as a daemon thread target. It:
    1. Imports the Flask app and initializes the database
    2. Creates required directories (uploads, reports)
    3. Suppresses verbose HTTP logs (we don't need them in desktop mode)
    4. Starts Flask on localhost with threading enabled
    
    Args:
        port (int): Port number to bind Flask to
    
    Note: 
        - host='127.0.0.1' ensures only local connections (security)  
        - threaded=True allows multiple simultaneous requests (SSE needs this)
        - use_reloader=False prevents Flask from spawning a child process
    """
    from app import app
    from modules.database import init_db
    init_db()                      # Create tables + seed default users
    os.makedirs('uploads', exist_ok=True)   # Evidence files go here
    os.makedirs('reports', exist_ok=True)   # PDF/JSON reports go here

    # Suppress Flask's per-request HTTP logging (clutters console)
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Start Flask — this call blocks until the thread is killed
    app.run(
        host='127.0.0.1',       # Localhost only (not exposed to network)
        port=port,               # Dynamic port from find_free_port()
        debug=False,             # No debug mode in production desktop app
        threaded=True,           # Allow concurrent requests (needed for SSE)
        use_reloader=False       # Don't restart on file changes
    )


def main():
    """
    Main entry point — sets up and launches the FORAX desktop application.
    
    Flow:
    1. Check pywebview is installed
    2. Find a free port
    3. Start Flask in background thread
    4. Wait for Flask to be ready
    5. Open native window with pywebview
    6. Try different GUI backends until one works
    """
    # ── Step 1: Verify pywebview is installed ──────────────────
    try:
        import webview
    except ImportError:
        print("\n[FORAX] ERROR: pywebview not installed!")
        print("  Run:  pip install pywebview[cef]")
        input("Press Enter to exit...")
        sys.exit(1)

    # ── Step 2: Find a free port ──────────────────────────────
    port = find_free_port()

    # ── Console banner (shown in terminal) ────────────────────
    print()
    print("=" * 52)
    print("  FORAX — Desktop Application")
    print("  Forensic Analysis & eXtraction Platform")
    print()
    print("  This is a DESKTOP app — not a web browser.")
    print("  A native window will open automatically.")
    print("=" * 52)
    print()

    # ── Step 3: Start Flask in a background daemon thread ─────
    # daemon=True means this thread dies when the main thread exits
    flask_thread = threading.Thread(
        target=start_flask,
        args=(port,),
        daemon=True
    )
    flask_thread.start()

    # ── Step 4: Wait for Flask to be ready ────────────────────
    print("[FORAX] Starting...")
    # Increase timeout to 60 seconds for slow HDDs/database migrations
    if not wait_for_server(port, timeout=60):
        print("[FORAX] ERROR: Internal engine failed to start!")
        input("Press Enter to exit...")
        sys.exit(1)
    print("[FORAX] Opening application window...")

    # ── Step 5: Create native desktop window ──────────────────
    # This is NOT a browser — it's a native OS window that renders HTML
    window = webview.create_window(
        title='FORAX — Forensic Analysis & eXtraction',  # Window title bar
        url=f'http://127.0.0.1:{port}/login',            # Initial page to load
        width=1280,              # Default window width
        height=850,              # Default window height
        min_size=(1024, 700),    # Minimum window size (can't go smaller)
        resizable=True,          # Allow window resizing
        text_select=True,        # Allow text selection (for copying SHA hashes etc)
        confirm_close=True,      # Show "Are you sure?" dialog on close
    )

    # ── Step 6: Try GUI backends in order of preference ───────
    # CEF is preferred, but cefpython currently doesn't support Python 3.13+.
    # In that case, skip CEF and use EdgeChromium first.
    if sys.version_info >= (3, 13):
        gui_order = ['edgechromium', 'mshtml']
    else:
        gui_order = ['cef', 'edgechromium', 'mshtml']
    started = False
    for gui in gui_order:
        try:
            print(f"[FORAX] Trying GUI backend: {gui}")
            webview.start(
                gui=gui,             # Which rendering engine to use
                debug=False,         # No developer tools
                private_mode=False,  # Allow cookies/sessions to persist
            )
            started = True
            break  # If webview.start() returns normally, window was closed cleanly
        except KeyboardInterrupt:
            print("\n[FORAX] Shutdown requested by user.")
            sys.exit(0)
        except Exception as e:
            print(f"[FORAX] {gui} failed: {e}")
            # Recreate window object for next backend attempt
            try:
                window = webview.create_window(
                    title='FORAX — Forensic Analysis & eXtraction',
                    url=f'http://127.0.0.1:{port}/login',
                    width=1280,
                    height=850,
                    min_size=(1024, 700),
                    resizable=True,
                    text_select=True,
                    confirm_close=True,
                )
            except:
                pass
            continue  # Try next backend

    # ── Step 7: Handle case where no backend worked ───────────
    if not started:
        print("\n[FORAX] ERROR: No GUI backend available!")
        print("  Install one of these:")
        print("    pip install pywebview[cef]     (recommended)")
        print("    pip install pywebview")
        print("  Or install Microsoft Edge WebView2 Runtime")
        input("Press Enter to exit...")
        sys.exit(1)

    print("\n[FORAX] Application closed.")


# ── Script Entry Point ─────────────────────────────────────────
# Only run main() when this file is executed directly
# (not when imported by another module)
if __name__ == '__main__':
    main()
