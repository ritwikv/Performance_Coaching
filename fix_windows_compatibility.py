#!/usr/bin/env python3
"""
Windows Compatibility Fix for HuggingFace Hub Symlink Issues
Resolves the all-MiniLM-L6-v2 model loading warnings on Windows
"""

import os
import sys
import subprocess
from pathlib import Path

def set_environment_variable():
    """Set the HuggingFace Hub symlink warning environment variable"""
    print("🔧 Setting HuggingFace Hub environment variable...")
    
    # Set for current session
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    print("✅ Environment variable set for current session")
    
    # Instructions for permanent setting
    print("\n📋 To set permanently:")
    print("Option 1 - Command Prompt:")
    print('setx HF_HUB_DISABLE_SYMLINKS_WARNING "1"')
    print("\nOption 2 - PowerShell:")
    print('[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS_WARNING", "1", "User")')
    print("\nOption 3 - System Properties:")
    print("1. Right-click 'This PC' → Properties → Advanced system settings")
    print("2. Click 'Environment Variables'")
    print("3. Under 'User variables', click 'New'")
    print("4. Variable name: HF_HUB_DISABLE_SYMLINKS_WARNING")
    print("5. Variable value: 1")

def check_developer_mode():
    """Check if Windows Developer Mode is enabled"""
    print("\n🔍 Checking Windows Developer Mode status...")
    
    try:
        # Check registry for developer mode
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock")
        value, _ = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevLicense")
        winreg.CloseKey(key)
        
        if value == 1:
            print("✅ Developer Mode is enabled")
            return True
        else:
            print("❌ Developer Mode is disabled")
            return False
    except (ImportError, FileNotFoundError, OSError):
        print("⚠️ Could not check Developer Mode status")
        return False

def enable_developer_mode_instructions():
    """Provide instructions for enabling Developer Mode"""
    print("\n📋 To enable Windows Developer Mode:")
    print("1. Open Settings (Windows + I)")
    print("2. Go to Update & Security → For developers")
    print("3. Select 'Developer mode'")
    print("4. Restart your computer")
    print("\nAlternatively:")
    print("1. Search for 'Developer settings' in Start menu")
    print("2. Turn on 'Developer Mode'")

def test_symlink_support():
    """Test if symlinks work in the current environment"""
    print("\n🧪 Testing symlink support...")
    
    test_dir = Path("test_symlink")
    test_file = test_dir / "test.txt"
    test_link = test_dir / "test_link.txt"
    
    try:
        # Create test directory and file
        test_dir.mkdir(exist_ok=True)
        test_file.write_text("test content")
        
        # Try to create symlink
        if test_link.exists():
            test_link.unlink()
        
        test_link.symlink_to(test_file)
        
        if test_link.is_symlink() and test_link.read_text() == "test content":
            print("✅ Symlinks work correctly")
            result = True
        else:
            print("❌ Symlinks not working properly")
            result = False
            
    except (OSError, PermissionError) as e:
        print(f"❌ Symlinks not supported: {e}")
        result = False
    
    finally:
        # Cleanup
        try:
            if test_link.exists():
                test_link.unlink()
            if test_file.exists():
                test_file.unlink()
            if test_dir.exists():
                test_dir.rmdir()
        except:
            pass
    
    return result

def test_huggingface_loading():
    """Test HuggingFace model loading with the fix"""
    print("\n🧪 Testing HuggingFace model loading...")
    
    try:
        # Set environment variable
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        # Try to import and load model
        from sentence_transformers import SentenceTransformer
        
        print("📥 Loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"✅ Model loaded successfully!")
        print(f"✅ Test embedding shape: {embedding.shape}")
        return True
        
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def create_batch_file():
    """Create a batch file to set environment variable and launch Streamlit"""
    batch_content = '''@echo off
echo Setting HuggingFace Hub environment variable...
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Launching Streamlit dashboard...
python -m streamlit run enhanced_streamlit_dashboard.py

pause
'''
    
    batch_file = Path("launch_streamlit_windows.bat")
    batch_file.write_text(batch_content)
    print(f"✅ Created batch file: {batch_file}")
    print("You can double-click this file to launch Streamlit with the fix applied")

def create_powershell_script():
    """Create a PowerShell script to set environment variable and launch Streamlit"""
    ps_content = '''# Set HuggingFace Hub environment variable
Write-Host "Setting HuggingFace Hub environment variable..." -ForegroundColor Green
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Launch Streamlit dashboard
Write-Host "Launching Streamlit dashboard..." -ForegroundColor Green
python -m streamlit run enhanced_streamlit_dashboard.py

Read-Host "Press Enter to exit"
'''
    
    ps_file = Path("launch_streamlit_windows.ps1")
    ps_file.write_text(ps_content)
    print(f"✅ Created PowerShell script: {ps_file}")
    print("Run with: powershell -ExecutionPolicy Bypass -File launch_streamlit_windows.ps1")

def main():
    """Main function to run all Windows compatibility fixes"""
    print("🪟 Windows Compatibility Fix for HuggingFace Hub")
    print("=" * 60)
    
    # Check if we're on Windows
    if os.name != 'nt':
        print("ℹ️ This script is designed for Windows systems")
        print("Your system appears to be Unix-like and should not have symlink issues")
        return
    
    print(f"🖥️ Operating System: {os.name}")
    print(f"🐍 Python Version: {sys.version}")
    print(f"📁 Current Directory: {os.getcwd()}")
    
    # Step 1: Set environment variable
    set_environment_variable()
    
    # Step 2: Check Developer Mode
    dev_mode_enabled = check_developer_mode()
    if not dev_mode_enabled:
        enable_developer_mode_instructions()
    
    # Step 3: Test symlink support
    symlinks_work = test_symlink_support()
    
    # Step 4: Test HuggingFace loading
    hf_works = test_huggingface_loading()
    
    # Step 5: Create launcher scripts
    print("\n🚀 Creating launcher scripts...")
    create_batch_file()
    create_powershell_script()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPATIBILITY SUMMARY")
    print("=" * 60)
    print(f"Environment Variable Set: ✅")
    print(f"Developer Mode Enabled: {'✅' if dev_mode_enabled else '❌'}")
    print(f"Symlinks Working: {'✅' if symlinks_work else '❌'}")
    print(f"HuggingFace Loading: {'✅' if hf_works else '❌'}")
    
    print("\n🎯 RECOMMENDATIONS:")
    
    if hf_works:
        print("✅ HuggingFace model loading works! You can now use:")
        print("   - streamlit run enhanced_streamlit_dashboard.py")
        print("   - Or double-click launch_streamlit_windows.bat")
    else:
        print("⚠️ HuggingFace model loading still has issues. Try:")
        print("   1. Use basic mode: python launch_basic.py")
        print("   2. Enable Developer Mode (see instructions above)")
        print("   3. Run as administrator")
    
    if not symlinks_work and not dev_mode_enabled:
        print("💡 For optimal performance, consider enabling Developer Mode")
        print("   This will enable symlinks and reduce disk usage")
    
    print("\n🔧 QUICK FIXES:")
    print("1. Environment Variable (Current session): Already set ✅")
    print("2. Batch File Launcher: launch_streamlit_windows.bat ✅")
    print("3. PowerShell Launcher: launch_streamlit_windows.ps1 ✅")
    print("4. Basic Mode (No ML): python launch_basic.py")

if __name__ == "__main__":
    main()

