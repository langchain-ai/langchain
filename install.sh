#!/bin/bash

# Ticket Broker Optimization System - Installation Script
# This script helps set up the system with all required dependencies

echo "🎫 Ticket Broker Optimization System - Installation"
echo "=" * 50

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

echo "✅ Found: $python_version"

# Check if pip is available
echo "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not available. Please install pip first."
    exit 1
fi
echo "✅ pip3 is available"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r ticket_broker/requirements.txt

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi
echo "✅ Python dependencies installed"

# Check for Chrome/Chromium
echo "Checking for Chrome/Chromium browser..."
if command -v google-chrome &> /dev/null; then
    echo "✅ Found Google Chrome"
elif command -v chromium-browser &> /dev/null; then
    echo "✅ Found Chromium"
elif command -v chromium &> /dev/null; then
    echo "✅ Found Chromium"
else
    echo "⚠️  Chrome/Chromium not found. Some web scraping features may not work."
    echo "   Please install Chrome or Chromium for full functionality."
fi

# Check for ChromeDriver
echo "Checking for ChromeDriver..."
if command -v chromedriver &> /dev/null; then
    echo "✅ ChromeDriver is available"
else
    echo "⚠️  ChromeDriver not found. Installing ChromeDriver..."
    
    # Detect OS and install ChromeDriver
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing ChromeDriver on Linux..."
        sudo apt-get update
        sudo apt-get install -y chromium-chromedriver
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing ChromeDriver via Homebrew..."
            brew install chromedriver
        else
            echo "❌ Homebrew not found. Please install ChromeDriver manually:"
            echo "   brew install chromedriver"
            echo "   or download from: https://chromedriver.chromium.org/"
        fi
    else
        echo "❌ Unsupported OS. Please install ChromeDriver manually:"
        echo "   Download from: https://chromedriver.chromium.org/"
    fi
fi

# Set up environment file
echo "Setting up environment configuration..."
if [[ ! -f "ticket_broker/.env" ]]; then
    cp ticket_broker/.env.example ticket_broker/.env
    echo "✅ Created .env file from template"
    echo "📝 Please edit ticket_broker/.env and add your API keys"
else
    echo "✅ .env file already exists"
fi

# Create reports directory
mkdir -p reports
echo "✅ Created reports directory"

# Test the installation
echo "Testing installation..."
cd ticket_broker
python3 -c "
try:
    from main import TicketBrokerOptimizer
    print('✅ System imports successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [[ $? -eq 0 ]]; then
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Edit ticket_broker/.env with your API keys:"
    echo "   - Spotify: https://developer.spotify.com/"
    echo "   - Twitter: https://developer.twitter.com/"
    echo "   - Ticketmaster: https://developer.ticketmaster.com/"
    echo ""
    echo "2. Run the example analysis:"
    echo "   python3 example_analysis.py"
    echo ""
    echo "3. Start analyzing events:"
    echo "   python3 -c \"from ticket_broker import run_example_analysis; run_example_analysis()\""
    echo ""
    echo "📖 Read the documentation in ticket_broker/README.md for detailed usage"
else
    echo "❌ Installation test failed. Please check the error messages above."
    exit 1
fi