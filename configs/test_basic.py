#!/usr/bin/env python3
"""
Basic Test Script for LDPC Steganography System
Run this script to verify that the system is working correctly
"""

import sys
import os
import torch
import numpy as np
import traceback

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    
    try:
        # Test PyTorch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA device: {torch.cuda.get_device_name()}")
        
        # Test NumPy
        print(f"  ✓ NumPy version: {np.__version__}")
        
        # Test project imports
        from configs.ldpc_config import LDPCConfig
        print("  ✓ Config import successful")
        
        from core.adaptive_ldpc import create_ldpc_system
        print("  ✓ LDPC core import successful")
        
        print("✅ All imports successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration system"""
    print("🔧 Testing configuration...")
    
    try:
        from configs.ldpc_config import LDPCConfig, setup_debug_config
        
        # Test basic config
        config = LDPCConfig()
        print(f"  ✓ Basic config created")
        print(f"    - Device: {config.device}")
        print(f"    - Message length: {config.message_length}")
        print(f"    - Image size: {config.image_size}")
        
        # Test debug config
        debug_config = setup_debug_config()
        print(f"  ✓ Debug config created")
        print(f"    - Device: {debug_config.device}")
        print(f"    - Batch size: {debug_config.batch_size}")
        
        print("✅ Configuration tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_ldpc_system():
    """Test LDPC system creation and basic operations"""
    print("🧮 Testing LDPC system...")
    
    try:
        from configs.ldpc_config import setup_debug_config
        from core.adaptive_ldpc import create_ldpc_system
        
        # Create debug config
        config = setup_debug_config()
        
        # Create LDPC system
        ldpc_system = create_ldpc_system(config)
        print("  ✓ LDPC system created")
        
        # Test encoding/decoding
        message = torch.randint(0, 2, (1, config.message_length), dtype=torch.float32)
        print(f"  ✓ Test message created: shape {message.shape}")
        
        # Encode
        encoded = ldpc_system.encode(message, attack_strength=0.3)
        print(f"  ✓ Message encoded: shape {encoded.shape}")
        
        # Decode
        decoded = ldpc_system.decode(encoded, attack_strength=0.3)
        print(f"  ✓ Message decoded: shape {decoded.shape}")
        
        # Check accuracy
        original_bits = (message > 0.5).float()
        decoded_bits = (decoded > 0.5).float()
        accuracy = (original_bits == decoded_bits).float().mean()
        print(f"  ✓ Decoding accuracy: {accuracy:.1%}")
        
        print("✅ LDPC system tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ LDPC system test failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("🤖 Testing model creation...")
    
    try:
        from configs.ldpc_config import setup_debug_config
        from core.adaptive_ldpc import create_ldpc_system
        from models.steganography_model import AdvancedSteganographyModelWithLDPC
        
        # Create config and LDPC system
        config = setup_debug_config()
        ldpc_system = create_ldpc_system(config)
        
        # Create model
        model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
        print("  ✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        
        print("✅ Model creation tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass through the model"""
    print("➡️ Testing forward pass...")
    
    try:
        from configs.ldpc_config import setup_debug_config
        from core.adaptive_ldpc import create_ldpc_system
        from models.steganography_model import AdvancedSteganographyModelWithLDPC
        
        # Create components
        config = setup_debug_config()
        ldpc_system = create_ldpc_system(config)
        model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
        
        # Create test inputs
        cover_images = torch.randn(1, config.channels, config.image_size, config.image_size)
        messages = torch.randint(0, 2, (1, config.message_length), dtype=torch.float32)
        
        print(f"  ✓ Test inputs created")
        print(f"    - Cover images: {cover_images.shape}")
        print(f"    - Messages: {messages.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(cover_images, messages)
        
        print(f"  ✓ Forward pass completed")
        print(f"    - Stego images: {outputs['stego_images'].shape}")
        print(f"    - Extracted messages: {outputs['extracted_messages'].shape}")
        
        # Check output validity
        assert outputs['stego_images'].shape == cover_images.shape, "Stego image shape mismatch"
        assert outputs['extracted_messages'].shape == messages.shape, "Message shape mismatch"
        
        # Calculate basic metrics
        mse = torch.mean((outputs['stego_images'] - cover_images) ** 2)
        print(f"  ✓ Image MSE: {mse:.6f}")
        
        # Message accuracy
        original_bits = (messages > 0.5).float()
        extracted_bits = (outputs['extracted_messages'] > 0).float()
        accuracy = (original_bits == extracted_bits).float().mean()
        print(f"  ✓ Message accuracy: {accuracy:.1%}")
        
        print("✅ Forward pass tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        traceback.print_exc()
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("📊 Testing synthetic data...")
    
    try:
        from data.datasets import SyntheticSteganographyDataset
        from torch.utils.data import DataLoader
        
        # Create synthetic dataset
        dataset = SyntheticSteganographyDataset(
            num_samples=10,
            image_size=64,
            message_length=128
        )
        print(f"  ✓ Synthetic dataset created: {len(dataset)} samples")
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print(f"  ✓ Data loader created")
        
        # Test batch loading
        batch = next(iter(dataloader))
        print(f"  ✓ Batch loaded:")
        print(f"    - Cover images: {batch['cover'].shape}")
        print(f"    - Messages: {batch['message'].shape}")
        print(f"    - Indices: {batch['index'].shape}")
        
        print("✅ Synthetic data tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 LDPC Steganography System - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("LDPC System", test_ldpc_system),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Synthetic Data", test_synthetic_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        print("-" * 30)
        
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} test failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("\nNext steps:")
        print("1. Try the introduction notebook: jupyter notebook notebooks/01_ldpc_introduction.ipynb")
        print("2. Run training: python experiments/train_ldpc.py")
        print("3. Explore the documentation in docs/")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements_ldpc.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Report issues at: https://github.com/your-username/ldpc-steganography/issues")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)