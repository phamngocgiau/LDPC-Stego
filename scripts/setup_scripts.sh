#!/bin/bash
# setup_environment.sh
# Environment setup script for LDPC Steganography System

set -e

echo "🚀 Setting up LDPC Steganography System Environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_header "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        print_status "Detected Windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_header "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_status "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8 or higher
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_header "Installing system dependencies..."
    
    case $OS in
        "linux")
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y build-essential python3-dev python3-pip python3-venv git wget curl
            elif command -v yum &> /dev/null; then
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y python3-devel python3-pip git wget curl
            elif command -v dnf &> /dev/null; then
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y python3-devel python3-pip git wget curl
            else
                print_warning "Could not detect package manager. Please install build tools manually."
            fi
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install git wget curl
            ;;
        "windows")
            print_warning "Please ensure you have Visual Studio Build Tools installed."
            print_warning "Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            ;;
    esac
}

# Setup virtual environment
setup_venv() {
    print_header "Setting up virtual environment..."
    
    VENV_NAME="ldpc-stego-env"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    python3 -m venv "$VENV_NAME"
    print_status "Created virtual environment: $VENV_NAME"
    
    # Activation instructions
    case $OS in
        "windows")
            ACTIVATE_CMD="$VENV_NAME\\Scripts\\activate"
            ;;
        *)
            ACTIVATE_CMD="source $VENV_NAME/bin/activate"
            ;;
    esac
    
    print_status "To activate the environment, run: $ACTIVATE_CMD"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python dependencies..."
    
    # Activate virtual environment
    case $OS in
        "windows")
            source "$VENV_NAME/Scripts/activate"
            ;;
        *)
            source "$VENV_NAME/bin/activate"
            ;;
    esac
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch (CPU version by default)
    print_status "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install other requirements
    if [ -f "requirements_ldpc.txt" ]; then
        print_status "Installing LDPC requirements..."
        pip install -r requirements_ldpc.txt
    elif [ -f "requirements.txt" ]; then
        print_status "Installing basic requirements..."
        pip install -r requirements.txt
    else
        print_error "No requirements file found!"
        exit 1
    fi
    
    # Install package in development mode
    print_status "Installing package in development mode..."
    pip install -e .
}

# Check GPU support
check_gpu() {
    print_header "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=gpu_name,memory.total --format=csv
        
        read -p "Would you like to install CUDA-enabled PyTorch? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Installing CUDA-enabled PyTorch..."
            pip uninstall -y torch torchvision torchaudio
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        print_warning "No NVIDIA GPU detected. Using CPU-only PyTorch."
    fi
}

# Setup directories
setup_directories() {
    print_header "Setting up project directories..."
    
    DIRS=("data/train" "data/val" "data/test" "results/runs" "results/models" "results/logs" "results/figures")
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
        print_status "Created directory: $dir"
    done
}

# Download sample data
download_sample_data() {
    print_header "Downloading sample data..."
    
    read -p "Would you like to download sample data for testing? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/download_data.py --sample-only
    else
        print_status "Skipping sample data download"
    fi
}

# Run tests
run_tests() {
    print_header "Running verification tests..."
    
    # Activate virtual environment
    case $OS in
        "windows")
            source "$VENV_NAME/Scripts/activate"
            ;;
        *)
            source "$VENV_NAME/bin/activate"
            ;;
    esac
    
    print_status "Running basic import test..."
    python -c "
import torch
import numpy as np
from configs.ldpc_config import LDPCConfig
from core.adaptive_ldpc import create_ldpc_system

print('✓ Basic imports successful')

config = LDPCConfig()
config.device = 'cpu'
config.message_length = 128
config.image_size = 64

ldpc_system = create_ldpc_system(config)
print('✓ LDPC system creation successful')

# Test encoding/decoding
message = torch.randint(0, 2, (1, 128), dtype=torch.float32)
encoded = ldpc_system.encode(message, attack_strength=0.3)
decoded = ldpc_system.decode(encoded, attack_strength=0.3)
print('✓ LDPC encoding/decoding successful')

print('🎉 All tests passed!')
"
    
    if [ $? -eq 0 ]; then
        print_status "✅ Verification tests passed!"
    else
        print_error "❌ Verification tests failed!"
        exit 1
    fi
}

# Print setup completion
print_completion() {
    print_header "Setup completed successfully! 🎉"
    
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   $ACTIVATE_CMD"
    echo
    echo "2. Explore the notebooks:"
    echo "   jupyter notebook notebooks/01_ldpc_introduction.ipynb"
    echo
    echo "3. Try a quick test:"
    echo "   python experiments/demo.py"
    echo
    echo "4. Read the documentation:"
    echo "   docs/README.md"
    echo
    echo "For more information, visit: https://github.com/your-username/ldpc-steganography"
}

# Main execution
main() {
    echo "🔧 LDPC Steganography System - Environment Setup"
    echo "================================================"
    echo
    
    check_os
    check_python
    install_system_deps
    setup_venv
    install_python_deps
    check_gpu
    setup_directories
    download_sample_data
    run_tests
    print_completion
}

# Handle interrupts
trap 'print_error "Setup interrupted!"; exit 1' INT

# Run main function
main "$@"