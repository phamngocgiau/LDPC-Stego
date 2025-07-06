#!/usr/bin/env python3
"""
Integration Tests for LDPC Steganography System
End-to-end testing of the complete system
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCConfig
from core.adaptive_ldpc import create_ldpc_system, AdaptiveLDPC
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from data.datasets import SyntheticSteganographyDataset
from torch.utils.data import DataLoader
from training.trainer import LDPCTrainer
from evaluation.evaluator import LDPCEvaluator


class IntegrationTestBase(unittest.TestCase):
    """Base class for integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Create minimal config
        cls.config = LDPCConfig()
        cls.config.device = 'cpu'
        cls.config.image_size = 64
        cls.config.channels = 3
        cls.config.message_length = 256
        cls.config.batch_size = 2
        cls.config.num_epochs = 2
        cls.config.unet_base_channels = 32
        cls.config.unet_depth = 3
        cls.config.attention_layers = [2]
        cls.config.ldpc_min_redundancy = 0.1
        cls.config.ldpc_max_redundancy = 0.3
        cls.config.output_dir = str(cls.temp_dir)
        cls.config.log_dir = str(cls.temp_dir / 'logs')
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(cls.temp_dir)


class TestSystemIntegration(IntegrationTestBase):
    """Test complete system integration"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # 1. Create LDPC system
        ldpc_system = create_ldpc_system(self.config)
        self.assertIsInstance(ldpc_system, AdaptiveLDPC)
        
        # 2. Create model
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        self.assertIsNotNone(model)
        
        # 3. Create dataset
        dataset = SyntheticSteganographyDataset(
            num_samples=10,
            image_size=self.config.image_size,
            message_length=self.config.message_length
        )
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        val_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        # 4. Test forward pass
        batch = next(iter(train_loader))
        cover_images = batch['cover']
        messages = batch['message']
        
        with torch.no_grad():
            outputs = model(cover_images, messages)
        
        # Check outputs
        self.assertIn('stego_images', outputs)
        self.assertIn('extracted_messages', outputs)
        self.assertEqual(outputs['stego_images'].shape, cover_images.shape)
    
    def test_training_integration(self):
        """Test training integration"""
        # Create components
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Create data loaders
        dataset = SyntheticSteganographyDataset(
            num_samples=8,
            image_size=self.config.image_size,
            message_length=self.config.message_length
        )
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        val_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        # Create trainer
        trainer = LDPCTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config
        )
        
        # Short training run
        history = trainer.train(num_epochs=1)
        
        # Check training completed
        self.assertIsInstance(history, dict)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
    
    def test_evaluation_integration(self):
        """Test evaluation integration"""
        # Create components
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Create test data
        dataset = SyntheticSteganographyDataset(
            num_samples=4,
            image_size=self.config.image_size,
            message_length=self.config.message_length
        )
        test_loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        # Create evaluator
        evaluator = LDPCEvaluator(
            model=model,
            ldpc_system=ldpc_system,
            test_loader=test_loader,
            config=self.config
        )
        
        # Run evaluation
        results = evaluator.evaluate_basic()
        
        # Check results
        self.assertIsInstance(results, dict)
        expected_keys = ['psnr', 'ssim', 'message_accuracy', 'bit_error_rate']
        for key in expected_keys:
            self.assertIn(key, results)


class TestLDPCSystemIntegration(IntegrationTestBase):
    """Test LDPC system integration"""
    
    def test_ldpc_creation_and_configuration(self):
        """Test LDPC system creation with different configurations"""
        # Test different redundancy levels
        configs = [
            {'min_redundancy': 0.1, 'max_redundancy': 0.3},
            {'min_redundancy': 0.2, 'max_redundancy': 0.5},
        ]
        
        for config_params in configs:
            config = LDPCConfig()
            config.message_length = 256
            config.device = 'cpu'
            config.ldpc_min_redundancy = config_params['min_redundancy']
            config.ldpc_max_redundancy = config_params['max_redundancy']
            
            ldpc_system = create_ldpc_system(config)
            
            # Test encoding/decoding
            messages = torch.randint(0, 2, (2, 256), dtype=torch.float32)
            encoded = ldpc_system.encode(messages, attack_strength=0.3)
            decoded = ldpc_system.decode(encoded, attack_strength=0.3)
            
            self.assertEqual(decoded.shape, messages.shape)
    
    def test_ldpc_performance_across_attacks(self):
        """Test LDPC performance across different attack strengths"""
        ldpc_system = create_ldpc_system(self.config)
        
        attack_strengths = [0.1, 0.3, 0.5]
        messages = torch.randint(0, 2, (5, self.config.message_length), dtype=torch.float32)
        
        for attack_strength in attack_strengths:
            # Encode
            encoded = ldpc_system.encode(messages, attack_strength=attack_strength)
            
            # Simulate channel noise
            noise_level = attack_strength * 0.2
            noisy_encoded = encoded + torch.randn_like(encoded) * noise_level
            
            # Decode
            decoded = ldpc_system.decode(noisy_encoded, attack_strength=attack_strength)
            
            # Check basic properties
            self.assertEqual(decoded.shape, messages.shape)
            
            # Calculate accuracy
            original_bits = (messages > 0.5).float()
            decoded_bits = (decoded > 0.5).float()
            accuracy = (original_bits == decoded_bits).float().mean()
            
            # Accuracy should be reasonable (at least 70% for these simple tests)
            self.assertGreater(accuracy.item(), 0.7)
    
    def test_adaptive_redundancy(self):
        """Test adaptive redundancy selection"""
        ldpc_system = create_ldpc_system(self.config)
        
        # Test different attack strengths
        attack_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for attack_strength in attack_strengths:
            redundancy = ldpc_system.calculate_adaptive_redundancy(attack_strength)
            
            # Check redundancy is within bounds
            self.assertGreaterEqual(redundancy, self.config.ldpc_min_redundancy)
            self.assertLessEqual(redundancy, self.config.ldpc_max_redundancy)
            
            # Check that higher attack strength leads to higher redundancy
            if attack_strength > 0:
                lower_redundancy = ldpc_system.calculate_adaptive_redundancy(0.0)
                self.assertGreaterEqual(redundancy, lower_redundancy)


class TestModelIntegration(IntegrationTestBase):
    """Test model integration"""
    
    def test_model_components_integration(self):
        """Test integration between model components"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Test different input sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            cover_images = torch.randn(batch_size, 3, 64, 64)
            messages = torch.randint(0, 2, (batch_size, self.config.message_length), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(cover_images, messages)
            
            # Check all required outputs
            required_outputs = ['stego_images', 'extracted_messages']
            for output_key in required_outputs:
                self.assertIn(output_key, outputs)
                self.assertEqual(outputs[output_key].shape[0], batch_size)
    
    def test_gradient_flow_integration(self):
        """Test gradient flow through the complete model"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Create inputs
        cover_images = torch.randn(2, 3, 64, 64, requires_grad=True)
        messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
        
        # Forward pass
        outputs = model(cover_images, messages)
        
        # Simple loss
        loss = torch.mean((outputs['stego_images'] - cover_images) ** 2)
        loss += torch.mean((outputs['extracted_messages'] - messages) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(cover_images.grad)
        
        # Check model parameters have gradients
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
        
        # Most parameters should have gradients
        gradient_ratio = grad_count / param_count
        self.assertGreater(gradient_ratio, 0.8)


class TestDataPipelineIntegration(IntegrationTestBase):
    """Test data pipeline integration"""
    
    def test_data_loading_integration(self):
        """Test data loading with model"""
        # Create dataset
        dataset = SyntheticSteganographyDataset(
            num_samples=10,
            image_size=self.config.image_size,
            message_length=self.config.message_length
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Test multiple batches
        batch_count = 0
        for batch in data_loader:
            batch_count += 1
            
            # Check batch structure
            self.assertIn('cover', batch)
            self.assertIn('message', batch)
            self.assertIn('index', batch)
            
            # Check shapes
            self.assertEqual(batch['cover'].shape[1:], (3, 64, 64))
            self.assertEqual(batch['message'].shape[1], self.config.message_length)
            
            if batch_count >= 3:  # Test first 3 batches
                break
    
    def test_different_message_lengths(self):
        """Test system with different message lengths"""
        message_lengths = [128, 256, 512]
        
        for msg_len in message_lengths:
            # Update config
            config = LDPCConfig()
            config.device = 'cpu'
            config.image_size = 64
            config.message_length = msg_len
            config.batch_size = 2
            config.unet_base_channels = 32
            config.ldpc_min_redundancy = 0.1
            config.ldpc_max_redundancy = 0.3
            
            # Create components
            ldpc_system = create_ldpc_system(config)
            model = AdvancedSteganographyModelWithLDPC(config, ldpc_system)
            
            # Test forward pass
            cover_images = torch.randn(2, 3, 64, 64)
            messages = torch.randint(0, 2, (2, msg_len), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(cover_images, messages)
            
            # Check output shapes are correct
            self.assertEqual(outputs['stego_images'].shape, cover_images.shape)
            self.assertEqual(outputs['extracted_messages'].shape[1], msg_len)


class TestConfigurationIntegration(IntegrationTestBase):
    """Test configuration system integration"""
    
    def test_config_serialization(self):
        """Test configuration saving and loading"""
        config_path = self.temp_dir / 'test_config.json'
        
        # Save config
        self.config.save(str(config_path))
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = LDPCConfig.load(str(config_path))
        
        # Compare key attributes
        self.assertEqual(loaded_config.message_length, self.config.message_length)
        self.assertEqual(loaded_config.image_size, self.config.image_size)
        self.assertEqual(loaded_config.batch_size, self.config.batch_size)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid config
        valid_config = LDPCConfig()
        valid_config.message_length = 256
        valid_config.image_size = 64
        valid_config.ldpc_min_redundancy = 0.1
        valid_config.ldpc_max_redundancy = 0.5
        
        # Should not raise exception
        ldpc_system = create_ldpc_system(valid_config)
        self.assertIsNotNone(ldpc_system)
        
        # Test invalid config
        invalid_config = LDPCConfig()
        invalid_config.message_length = 0  # Invalid
        
        # Should handle gracefully
        with self.assertRaises((ValueError, RuntimeError)):
            ldpc_system = create_ldpc_system(invalid_config)


class TestRobustnessIntegration(IntegrationTestBase):
    """Test system robustness"""
    
    def test_noise_robustness(self):
        """Test system robustness to noise"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Create clean inputs
        cover_images = torch.randn(2, 3, 64, 64)
        messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
        
        # Add different levels of noise
        noise_levels = [0.0, 0.01, 0.05, 0.1]
        
        for noise_level in noise_levels:
            noisy_images = cover_images + torch.randn_like(cover_images) * noise_level
            
            with torch.no_grad():
                outputs = model(noisy_images, messages)
            
            # Check outputs are still valid
            self.assertFalse(torch.isnan(outputs['stego_images']).any())
            self.assertFalse(torch.isnan(outputs['extracted_messages']).any())
    
    def test_edge_cases(self):
        """Test edge cases"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Test with all zeros
        cover_zeros = torch.zeros(1, 3, 64, 64)
        message_zeros = torch.zeros(1, self.config.message_length)
        
        with torch.no_grad():
            outputs = model(cover_zeros, message_zeros)
        
        self.assertFalse(torch.isnan(outputs['stego_images']).any())
        
        # Test with all ones
        cover_ones = torch.ones(1, 3, 64, 64)
        message_ones = torch.ones(1, self.config.message_length)
        
        with torch.no_grad():
            outputs = model(cover_ones, message_ones)
        
        self.assertFalse(torch.isnan(outputs['stego_images']).any())


class TestPerformanceIntegration(IntegrationTestBase):
    """Test performance integration"""
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Track memory usage over multiple iterations
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for i in range(5):
            cover_images = torch.randn(2, 3, 64, 64)
            messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(cover_images, messages)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory shouldn't grow too much (allowing for some variance)
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            self.assertLess(memory_growth, 100 * 1024 * 1024)  # Less than 100MB growth
    
    def test_batch_size_scaling(self):
        """Test performance with different batch sizes"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            cover_images = torch.randn(batch_size, 3, 64, 64)
            messages = torch.randint(0, 2, (batch_size, self.config.message_length), dtype=torch.float32)
            
            # Time the forward pass
            import time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(cover_images, messages)
            
            end_time = time.time()
            
            # Check reasonable performance (should complete in under 10 seconds)
            processing_time = end_time - start_time
            self.assertLess(processing_time, 10.0)


class TestErrorHandlingIntegration(IntegrationTestBase):
    """Test error handling integration"""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Test wrong image dimensions
        with self.assertRaises((RuntimeError, ValueError)):
            wrong_size_images = torch.randn(2, 3, 32, 32)  # Wrong size
            messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
            model(wrong_size_images, messages)
        
        # Test wrong message length
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            cover_images = torch.randn(2, 3, 64, 64)
            wrong_messages = torch.randint(0, 2, (2, 100), dtype=torch.float32)  # Wrong length
            model(cover_images, wrong_messages)
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Move model to CUDA
        model = model.cuda()
        
        # Test with CPU inputs (should either work or give clear error)
        cover_images = torch.randn(1, 3, 64, 64)  # CPU tensor
        messages = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
        
        with self.assertRaises(RuntimeError):
            model(cover_images, messages)


class TestSystemStressTest(IntegrationTestBase):
    """Stress test the complete system"""
    
    def test_continuous_operation(self):
        """Test continuous operation over many iterations"""
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        # Run many iterations to check for memory leaks, crashes, etc.
        for i in range(20):
            cover_images = torch.randn(1, 3, 64, 64)
            messages = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
            
            with torch.no_grad():
                outputs = model(cover_images, messages)
            
            # Check outputs are still valid
            self.assertFalse(torch.isnan(outputs['stego_images']).any())
            self.assertFalse(torch.isnan(outputs['extracted_messages']).any())
            
            # Periodic cleanup
            if i % 5 == 0:
                import gc
                gc.collect()
    
    def test_concurrent_processing(self):
        """Test concurrent processing (threading safety)"""
        import threading
        import queue
        
        ldpc_system = create_ldpc_system(self.config)
        model = AdvancedSteganographyModelWithLDPC(self.config, ldpc_system)
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                cover_images = torch.randn(1, 3, 64, 64)
                messages = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = model(cover_images, messages)
                
                results_queue.put(outputs)
            except Exception as e:
                errors_queue.put(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertTrue(errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}")
        self.assertEqual(results_queue.qsize(), 3)


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)