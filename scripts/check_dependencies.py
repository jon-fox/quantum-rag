#!/usr/bin/env python3
"""
Script to check if all dependencies are properly installed in the current environment.
"""
import importlib
import sys
import os
import subprocess

def check_module(module_name):
    """Attempt to import a module and return True if successful"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Main function to check all dependencies"""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("-" * 50)

    print("Checking environment variables:")
    print(f"OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    print(f"HUGGINGFACE_API_TOKEN set: {'Yes' if os.environ.get('HUGGINGFACE_API_TOKEN') else 'No'}")
    print("-" * 50)

    # Check core packages
    packages = [
        "langchain", 
        "langchain_community",
        "langchain_openai", 
        "langchain_huggingface",
        "openai",
        "sentence_transformers", 
        "faiss",
        "numpy",
        "qiskit"
    ]
    
    print("Checking if packages are installed:")
    for package in packages:
        installed = check_module(package)
        print(f"{package}: {'Installed' if installed else 'Not Installed'}")
    
    print("-" * 50)
    
    # Get installed packages versions
    print("Package versions:")
    try:
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8').split('\n')
        for package in packages:
            for installed_package in installed_packages:
                if installed_package.lower().startswith(package.lower()):
                    print(installed_package)
                    break
    except Exception as e:
        print(f"Error getting package versions: {e}")
    
if __name__ == "__main__":
    main()
