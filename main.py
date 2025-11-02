#!/usr/bin/env python3
"""
Text Processing Application
Uses Hugging Face models to:
- Extract and summarize text
- Generate keywords
- Create test questions with multiple choice answers
"""

import sys
from ui import ConsoleUI
from config import Config

def check_environment():
    """Check if environment is properly configured"""
    try:
        Config.validate()
        return True
    except ValueError as e:
        print("\n" + "="*60)
        print("⚠️  CONFIGURATION ERROR")
        print("="*60)
        print(f"\n{str(e)}")
        print("\nPlease make sure to:")
        print("1. Create a .env file in the project directory")
        print("2. Add your Hugging Face API token:")
        print("   HUGGINGFACE_API_TOKEN=your_token_here")
        print("\nYou can get a free token at: https://huggingface.co/settings/tokens")
        print("="*60 + "\n")
        return False

def main():
    """Main entry point"""
    print("\n🚀 Starting Text Processing Application...")
    
    # Check environment configuration
    if not check_environment():
        sys.exit(1)
    
    # Run the application
    try:
        ui = ConsoleUI()
        ui.run()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()