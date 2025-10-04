#!/usr/bin/env python3
"""
QBES Website Launcher
Easy launcher for the QBES website with integrated testing
"""

import os
import sys
import webbrowser
import time
import subprocess
from pathlib import Path

def main():
    """Launch the QBES website."""
    
    print("=" * 60)
    print("🚀 QBES Website Launcher")
    print("=" * 60)
    print("Quantum Biological Environment Simulator")
    print("Interactive Website with Live Testing")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('website'):
        print("❌ Error: website directory not found")
        print("Please run this script from the QBES root directory")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    
    required_packages = ['flask', 'flask_cors']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please install manually: pip install flask flask-cors")
            sys.exit(1)
    
    # Check QBES installation
    print("\n🧪 Checking QBES installation...")
    try:
        import qbes
        print(f"  ✅ QBES version {qbes.__version__}")
    except ImportError as e:
        print(f"  ⚠️  QBES import issue: {e}")
        print("  Website will work with limited functionality")
    
    # Start the server
    print("\n🌐 Starting website server...")
    
    # Change to website directory
    os.chdir('website')
    
    # Import and start server
    try:
        sys.path.insert(0, os.getcwd())
        from server import app
        
        host = '127.0.0.1'
        port = 5000
        url = f'http://{host}:{port}'
        
        print(f"🎯 Server starting at: {url}")
        print("\n📋 Available features:")
        print("  • Interactive quantum mechanics tutorial")
        print("  • Live QBES project demonstration")
        print("  • Real-time testing and validation")
        print("  • Interactive parameter simulation")
        print("  • Comprehensive project documentation")
        
        print(f"\n🔗 Opening browser to: {url}")
        print("   (If browser doesn't open, copy the URL above)")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("\n" + "=" * 60)
        print("🎉 QBES Website is now running!")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start Flask server
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        print("Thank you for using QBES!")
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure port 5000 is not in use")
        print("2. Check that all dependencies are installed")
        print("3. Verify QBES is properly installed")
        sys.exit(1)

if __name__ == '__main__':
    main()