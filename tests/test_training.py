#!/usr/bin/env python3
"""
Training Tests for LDPC Steganography System
Test training loop, loss functions, and optimization
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.ldpc_config import LDPCConfig
from core.adaptive_ldpc import create_ldpc_system
from models.steganography_model import AdvancedSteganographyModelWithLDPC
from training.trainer import LDPCTrainer
from training.losses.steganography_loss import CompositeSteganographyLoss
from training.metrics.evaluation_metrics import SteganographyMetrics
from data.datasets import SyntheticSteganographyDataset
from torch.utils.data import DataLoader


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        # Basic settings
        self.device = 'cpu'
        self.image_size = 64
        self.channels = 3
        self.message_length = 256
        self.batch_size = 2
        
        # Model settings
        self.unet_base_channels = 32
        self.unet_depth = 3
        self.attention_layers = [2]
        self.latent_dim = 128
        
        # LDPC settings
        self.ldpc_min_redundancy = 0.1
        self.ldpc_max_redundancy = 0.3
        self.ldpc_use_neural_decoder = False
        self.ldpc_parallel_encoder = False
        
        # Training settings
        self.learning_rate = 1e-3
        self.num_epochs = 2
        self.validation_frequency = 1
        self.save_frequency = 1
        self.clip_grad_norm = 1.0
        
        # Loss weights
        self.loss_weights = {
            'message': 10.0,
            'mse': 2.0,
            'lpips': 1.0,
            'ssim': 1.0,
            'adversarial': 0.5,
            'recovery_mse': 1.0,
            'recovery_kl': 0.1,
        }
        
        # Paths
        self.output_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
    def get(self, key, default=None):
        return getattr(self, key, default)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.message_length = 256
        
        # Create loss function
        self.loss_fn = CompositeSteganographyLoss(self.config)
        
    def test_composite_loss(self):
        """Test composite steganography loss"""
        # Create mock data
        cover_images = torch.randn(self.batch_size, self.channels, self.height, self.width)
        stego_images = torch.randn(self.batch_size, self.channels, self.height, self.width)
        recovered_images = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        original_messages = torch.randint(0, 2, (self.batch_size, self.message_length), dtype=torch.float32)
        extracted_messages = torch.randn(self.batch_size, self.message_length)
        
        # Mock model outputs
        outputs = {
            'stego_images': stego_images,
            'extracted_messages': extracted_messages,
            'recovered_images': recovered_images,
            'recovery_mu': torch.randn(self.batch_size, 128),
            'recovery_logvar': torch.randn(self.batch_size, 128)
        }
        
        # Compute loss
        loss_dict = self.loss_fn(outputs, cover_images, original_messages)
        
        # Check that all loss components are present
        expected_keys = ['total_loss', 'message_loss', 'mse_loss', 'ssim_loss', 'recovery_loss']
        for key in expected_keys:
            self.assertIn(key, loss_dict)
            self.assertIsInstance(loss_dict[key], torch.Tensor)
            self.assertFalse(torch.isnan(loss_dict[key]))
    
    def test_message_loss(self):
        """Test message reconstruction loss"""
        original = torch.randint(0, 2, (self.batch_size, self.message_length), dtype=torch.float32)
        extracted = torch.randn(self.batch_size, self.message_length)
        
        loss = self.loss_fn.message_loss(extracted, original)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_perceptual_loss(self):
        """Test perceptual loss computation"""
        img1 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        img2 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # This might fail if LPIPS is not available, so we catch the exception
        try:
            loss = self.loss_fn.perceptual_loss(img1, img2)
            self.assertIsInstance(loss, torch.Tensor)
        except Exception as e:
            self.skipTest(f"Perceptual loss not available: {e}")


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def setUp(self):
        self.config = MockConfig()
        self.metrics = SteganographyMetrics(device='cpu')
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        self.message_length = 256
    
    def test_psnr_calculation(self):
        """Test PSNR calculation"""
        img1 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        img2 = img1 + torch.randn_like(img1) * 0.1  # Add some noise
        
        psnr = self.metrics.calculate_psnr(img1, img2)
        
        self.assertIsInstance(psnr, torch.Tensor)
        self.assertGreater(psnr.mean().item(), 0)
    
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        img1 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        img2 = img1 + torch.randn_like(img1) * 0.1
        
        ssim = self.metrics.calculate_ssim(img1, img2)
        
        self.assertIsInstance(ssim, torch.Tensor)
        self.assertGreaterEqual(ssim.mean().item(), 0)
        self.assertLessEqual(ssim.mean().item(), 1)
    
    def test_message_accuracy(self):
        """Test message accuracy calculation"""
        original = torch.randint(0, 2, (self.batch_size, self.message_length), dtype=torch.float32)
        extracted = torch.randn(self.batch_size, self.message_length)
        
        accuracy = self.metrics.calculate_message_accuracy(extracted, original)
        
        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertGreaterEqual(accuracy.item(), 0)
        self.assertLessEqual(accuracy.item(), 1)
    
    def test_bit_error_rate(self):
        """Test bit error rate calculation"""
        original = torch.randint(0, 2, (self.batch_size, self.message_length), dtype=torch.float32)
        extracted = torch.randn(self.batch_size, self.message_length)
        
        ber = self.metrics.calculate_bit_error_rate(extracted, original)
        
        self.assertIsInstance(ber, torch.Tensor)
        self.assertGreaterEqual(ber.item(), 0)
        self.assertLessEqual(ber.item(), 1)


class TestTrainingLoop(unittest.TestCase):
    """Test training loop components"""
    
    def setUp(self):
        self.config = MockConfig()
        
        # Create LDPC system
        self.ldpc_system = create_ldpc_system(self.config)
        
        # Create model
        self.model = AdvancedSteganographyModelWithLDPC(self.config, self.ldpc_system)
        
        # Create dataset and loader
        self.dataset = SyntheticSteganographyDataset(
            num_samples=10,
            image_size=self.config.image_size,
            message_length=self.config.message_length
        )
        
        self.train_loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False)
        
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = LDPCTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config
        )
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.loss_fn)
    
    def test_single_training_step(self):
        """Test single training step"""
        trainer = LDPCTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config
        )
        
        # Get a batch
        batch = next(iter(self.train_loader))
        
        # Training step
        loss_dict = trainer.training_step(batch, 0)
        
        # Check outputs
        self.assertIsInstance(loss_dict, dict)
        self.assertIn('total_loss', loss_dict)
        self.assertIsInstance(loss_dict['total_loss'], torch.Tensor)
    
    def test_validation_step(self):
        """Test validation step"""
        trainer = LDPCTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config
        )
        
        # Validation step
        val_metrics = trainer.validate()
        
        # Check outputs
        self.assertIsInstance(val_metrics, dict)
        expected_keys = ['val_loss', 'val_psnr', 'val_ssim', 'val_message_accuracy']
        for key in expected_keys:
            self.assertIn(key, val_metrics)
    
    def test_short_training_run(self):
        """Test short training run"""
        trainer = LDPCTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config
        )
        
        # Train for 1 epoch
        history = trainer.train(num_epochs=1)
        
        # Check history
        self.assertIsInstance(history, dict)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 1)


class TestOptimization(unittest.TestCase):
    """Test optimization components"""
    
    def setUp(self):
        self.config = MockConfig()
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        from training.trainer import create_optimizer
        
        optimizer = create_optimizer(self.model, self.config)
        
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
    
    def test_scheduler_creation(self):
        """Test scheduler creation"""
        from training.trainer import create_scheduler
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, self.config)
        
        # Should not be None for supported schedulers
        # May be None for unsupported schedulers
        if scheduler is not None:
            self.assertTrue(hasattr(scheduler, 'step'))
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        # Create dummy loss
        x = torch.randn(5, 10, requires_grad=True)
        y = torch.randn(5, 1)
        
        pred = self.model(x)
        loss = nn.MSELoss()(pred, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        total_norm_before = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        
        # Check gradients are clipped
        total_norm_after = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        if total_norm_before > max_norm:
            self.assertLessEqual(total_norm_after, max_norm + 1e-6)


class TestCheckpointing(unittest.TestCase):
    """Test model checkpointing"""
    
    def setUp(self):
        self.config = MockConfig()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint"""
        from utils.helpers import save_checkpoint
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = self.temp_dir / 'test_checkpoint.pth'
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.5,
            path=str(checkpoint_path),
            config=self.config
        )
        
        self.assertTrue(checkpoint_path.exists())
    
    def test_load_checkpoint(self):
        """Test loading checkpoint"""
        from utils.helpers import save_checkpoint, load_checkpoint
        
        # Create and save model
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = self.temp_dir / 'test_checkpoint.pth'
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.5,
            path=str(checkpoint_path),
            config=self.config
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(str(checkpoint_path), device='cpu')
        
        # Check contents
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('epoch', checkpoint)
        self.assertEqual(checkpoint['epoch'], 5)


class TestLDPCIntegration(unittest.TestCase):
    """Test LDPC integration in training"""
    
    def setUp(self):
        self.config = MockConfig()
        self.ldpc_system = create_ldpc_system(self.config)
        
    def test_ldpc_encoding_in_training(self):
        """Test LDPC encoding during training"""
        messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
        
        # Encode with LDPC
        encoded = self.ldpc_system.encode(messages, attack_strength=0.2)
        
        # Check output shape
        self.assertEqual(encoded.shape[0], 2)
        self.assertGreater(encoded.shape[1], self.config.message_length)
    
    def test_ldpc_decoding_in_training(self):
        """Test LDPC decoding during training"""
        messages = torch.randint(0, 2, (2, self.config.message_length), dtype=torch.float32)
        
        # Encode
        encoded = self.ldpc_system.encode(messages, attack_strength=0.2)
        
        # Add some noise (simulating extraction errors)
        noisy_encoded = encoded + torch.randn_like(encoded) * 0.1
        
        # Decode
        decoded = self.ldpc_system.decode(noisy_encoded, attack_strength=0.2)
        
        # Check output shape
        self.assertEqual(decoded.shape, messages.shape)
    
    def test_end_to_end_with_ldpc(self):
        """Test end-to-end training with LDPC"""
        model = AdvancedSteganographyModelWithLDPC(self.config, self.ldpc_system)
        
        # Create inputs
        cover_images = torch.randn(1, 3, 64, 64)
        messages = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
        
        # Forward pass
        outputs = model(cover_images, messages)
        
        # Check outputs
        self.assertIn('stego_images', outputs)
        self.assertIn('extracted_messages', outputs)
        self.assertEqual(outputs['stego_images'].shape, cover_images.shape)


if __name__ == '__main__':
    unittest.main()