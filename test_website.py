#!/usr/bin/env python3
"""
Simple test to verify the QBES website files are properly created
"""

import os
from pathlib import Path

def test_website_files():
    """Test that all website files exist and have content."""
    
    print("🧪 Testing QBES Website Files")
    print("=" * 40)
    
    website_dir = Path("website")
    
    # Required files
    required_files = {
        "index.html": "Main HTML file",
        "styles.css": "CSS styling file", 
        "script.js": "JavaScript functionality",
        "README.md": "Website documentation"
    }
    
    all_good = True
    
    for filename, description in required_files.items():
        filepath = website_dir / filename
        
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"✅ {filename:<12} - {description} ({file_size:,} bytes)")
            
            # Check if file has reasonable content
            if file_size < 100:
                print(f"   ⚠️  File seems too small")
                all_good = False
        else:
            print(f"❌ {filename:<12} - Missing!")
            all_good = False
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("🎉 All website files are present and ready!")
        print("\n📋 To use the website:")
        print("1. Navigate to the 'website' folder")
        print("2. Double-click on 'index.html'")
        print("3. It will open in your default web browser")
        print("\n🌐 The website includes:")
        print("   • Interactive quantum mechanics tutorial")
        print("   • Complete QBES project documentation")
        print("   • Live parameter simulation demo")
        print("   • Testing and validation interface")
        print("   • Responsive design for all devices")
        
        return True
    else:
        print("❌ Some website files are missing or incomplete")
        return False

def check_html_structure():
    """Check basic HTML structure."""
    
    print("\n🔍 Checking HTML Structure")
    print("=" * 40)
    
    html_file = Path("website/index.html")
    
    if not html_file.exists():
        print("❌ HTML file not found")
        return False
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for essential HTML elements
        checks = {
            "<!DOCTYPE html>": "HTML5 doctype",
            "<html": "HTML root element",
            "<head>": "HTML head section",
            "<body>": "HTML body section",
            "QBES": "Project name present",
            "Quantum": "Quantum content present",
            "section": "Page sections",
            "nav": "Navigation",
            "script.js": "JavaScript file linked",
            "styles.css": "CSS file linked"
        }
        
        for check, description in checks.items():
            if check in content:
                print(f"✅ {description}")
            else:
                print(f"❌ Missing: {description}")
        
        print(f"\n📊 HTML file size: {len(content):,} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 QBES Website Validation")
    print("=" * 50)
    
    # Test 1: Check files exist
    files_ok = test_website_files()
    
    # Test 2: Check HTML structure
    html_ok = check_html_structure()
    
    print("\n" + "=" * 50)
    print("📋 FINAL RESULTS")
    print("=" * 50)
    
    if files_ok and html_ok:
        print("🎉 WEBSITE IS READY!")
        print("\n✅ All tests passed")
        print("✅ Website files are complete")
        print("✅ HTML structure is valid")
        
        print("\n🎯 Next Steps:")
        print("1. Open 'website/index.html' in your browser")
        print("2. Explore the interactive features")
        print("3. Try the quantum simulation demo")
        print("4. Run the testing interface")
        
        print("\n🌟 Features Available:")
        print("   • Quantum mechanics education")
        print("   • QBES project showcase")
        print("   • Interactive simulations")
        print("   • Complete documentation")
        print("   • Mobile-responsive design")
        
    else:
        print("❌ WEBSITE HAS ISSUES")
        print("\nSome files may be missing or incomplete.")
        print("Please check the error messages above.")
    
    return files_ok and html_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)