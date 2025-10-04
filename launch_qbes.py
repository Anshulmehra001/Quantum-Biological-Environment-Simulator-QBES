#!/usr/bin/env python3
"""
QBES Launcher - Easy way to start using QBES
"""

import os
import sys
import webbrowser
import time
import subprocess
from pathlib import Path

def main():
    """Launch QBES with user-friendly interface."""
    
    print("🧬 QBES - Quantum Biological Environment Simulator")
    print("=" * 50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. 📚 Learn about QBES (Educational Website)")
        print("2. 🚀 Use QBES (Interactive Web Interface)")
        print("3. 💻 Command Line Demo")
        print("4. 📖 Read Documentation")
        print("5. 🧪 Run Tests")
        print("6. ❌ Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            launch_educational_website()
        elif choice == "2":
            launch_web_interface()
        elif choice == "3":
            run_command_demo()
        elif choice == "4":
            show_documentation()
        elif choice == "5":
            run_tests()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

def launch_educational_website():
    """Launch the educational website."""
    print("\n📚 Opening Educational Website...")
    website_path = Path("website/qbes_website.html")
    
    if website_path.exists():
        try:
            webbrowser.open(f"file://{website_path.absolute()}")
            print("✅ Educational website opened in your browser!")
            print("🎯 This website teaches you about quantum mechanics in biology")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print(f"📁 Manually open: {website_path.absolute()}")
    else:
        print("❌ Educational website not found!")
        print("📁 Expected location: website/qbes_website.html")

def launch_web_interface():
    """Launch the QBES web interface."""
    print("\n🚀 Starting QBES Web Interface...")
    
    try:
        # Check if Flask is available
        import flask
        print("✅ Flask is available")
        
        # Start the web interface
        print("🌐 Starting web server...")
        print("📍 Open your browser to: http://localhost:8080")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Open browser after a delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open("http://localhost:8080")
            except:
                pass
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the web interface
        from qbes.web_interface import main as web_main
        web_main()
        
    except ImportError:
        print("❌ Flask not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors'])
            print("✅ Flask installed! Please try again.")
        except:
            print("❌ Could not install Flask automatically.")
            print("💡 Please run: pip install flask flask-cors")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def run_command_demo():
    """Run command line demo."""
    print("\n💻 Running QBES Command Line Demo...")
    
    try:
        # Run the demo
        result = subprocess.run([sys.executable, "demo_qbes.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n⚠️ Demo completed with some issues.")
            
    except FileNotFoundError:
        print("❌ Demo file not found!")
        print("📁 Expected: demo_qbes.py")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def show_documentation():
    """Show available documentation."""
    print("\n📖 Available Documentation:")
    print("-" * 30)
    
    docs = [
        ("WHAT_IS_QBES.md", "Simple explanation of what QBES does"),
        ("HOW_TO_USE_QBES.md", "Complete usage guide"),
        ("README.md", "Project overview and quick start"),
        ("docs/user_guide.md", "Detailed user guide"),
        ("docs/tutorial.md", "Step-by-step tutorial"),
        ("docs/api_reference.md", "API documentation"),
    ]
    
    for doc_file, description in docs:
        if Path(doc_file).exists():
            print(f"✅ {doc_file:<25} - {description}")
        else:
            print(f"❌ {doc_file:<25} - Not found")
    
    print("\n💡 Tip: Open these files in any text editor to read them!")

def run_tests():
    """Run QBES tests."""
    print("\n🧪 Running QBES Tests...")
    
    tests = [
        ("demo_qbes.py", "Core functionality demo"),
        ("run_benchmarks.py", "Benchmark suite"),
        ("test_benchmark_final.py", "Benchmark validation"),
    ]
    
    for test_file, description in tests:
        if Path(test_file).exists():
            print(f"\n🔍 Running {description}...")
            try:
                result = subprocess.run([sys.executable, test_file], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"✅ {description} - PASSED")
                else:
                    print(f"⚠️ {description} - Issues detected")
                    if result.stderr:
                        print(f"   Error: {result.stderr[:200]}...")
                        
            except subprocess.TimeoutExpired:
                print(f"⏰ {description} - Timeout (taking too long)")
            except Exception as e:
                print(f"❌ {description} - Error: {e}")
        else:
            print(f"❌ {test_file} not found")

if __name__ == "__main__":
    main()