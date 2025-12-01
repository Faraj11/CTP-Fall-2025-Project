#!/usr/bin/env python3
"""Start Flask app with error handling"""

import sys
import traceback

try:
    print("=" * 60)
    print("Starting Flask Application...")
    print("=" * 60)
    
    from app import app
    
    print("\n[OK] App imported successfully")
    print("[OK] Starting Flask server on http://127.0.0.1:5000")
    print("=" * 60)
    print("\nServer is running! Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
    
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n[ERROR] Error starting Flask app: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
