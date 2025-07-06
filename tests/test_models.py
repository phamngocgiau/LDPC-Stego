#!/usr/bin/env python3
"""
Unit Tests for Neural Network Models
Test encoder, decoder, discriminator, and recovery networks
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoders.ldpc_aware_encoder import LDPCAwareDualUNetEncoder
from models.decoders.ldpc_aware_decoder import LDPCAwareDualUNetDecoder
from models.decoders.dual_unet_decoder import DualUNetDecoder
from models.recovery.recovery_cvae import RecoveryCVAE, AdaptiveRecoveryCVAE
from models.blocks.conv_block import ConvBlock, DepthwiseConvBlock
from models.blocks.residual_block import ResidualBlock
from models.attention.self_attention import SelfAttention, MultiScaleSelfAttention
from models.attention.cross_attention import CrossAttention
from configs.ldpc_config import LDPCConfig


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.image_size = 64  # Smaller for faster testing
        self.channels = 3
        self.message_length = 256
        self.unet_base_channels = 32
        self.unet_depth = 3
        self.attention_layers = [2, 4]
        self.device = 'cpu'
        self.latent_dim = 128
        self.ldpc_max_redundancy = 0.5
        self.ldpc_min_redundancy = 0.1
        
    def get(self, key, default=None):
        return getattr(self, key, default)


class TestConvBlocks(unittest.TestCase):
    """Test convolution blocks"""
    
    def setUp(self):
        self.batch_size = 2
        self.channels = 32
        self.height = 16
        self.width = 16
    
    def test_conv_block(self):
        """Test basic convolution block"""
        block = ConvBlock(self.channels, self.channels * 2)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.channels * 2, self.height, self.width))
        self.assertFalse(torch.isnan(output).any())
    
    def test_depthwise_conv_block(self):
        """Test depthwise separable convolution"""
        block = DepthwiseConvBlock(self.channels, self.channels * 2)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.channels * 2, self.height, self.width))
        
    def test_different_activations(self):
        """Test different activation functions"""
        activations = ['relu', 'gelu', 'silu', 'leaky_relu']
        
        for activation in activations:
            with self.subTest(activation=activation):
                block = ConvBlock(self.channels, self.channels, activation=activation)
                x = torch.randn(self.batch_size, self.channels, self.height, self.width)
                output = block(x)
                
                self.assertEqual(output.shape, x.shape)
                self.assertFalse(torch.isnan(output).any())
    
    def test_different_normalizations(self):
        """Test different normalization types"""
        norm_types = ['batch', 'group', 'layer', 'instance']
        
        for norm_type in norm_types:
            with self.subTest(norm_type=norm_type):
                block = ConvBlock(self.channels, self.channels, norm_type=norm_type)
                x = torch.randn(self.batch_size, self.channels, self.height, self.width)
                output = block(x)
                
                self.assertEqual(output.shape, x.shape)


class TestResidualBlocks(unittest.TestCase):
    """Test residual blocks"""
    
    def setUp(self):
        self.batch_size = 2
        self.channels = 32
        self.height = 16
        self.width = 16
    
    def test_residual_block(self):
        """Test basic residual block"""
        block = ResidualBlock(self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = block(x)
        
        self.assertEqual(output.shape, x.shape)
        
        # Check residual connection
        # Output should be different from input but maintain similar scale
        self.assertFalse(torch.allclose(output, x))
        self.assertLess(torch.mean(torch.abs(output - x)), torch.mean(torch.abs(x)))
    
    def test_residual_with_attention(self):
        """Test residual block with attention"""
        block = ResidualBlock(self.channels, use_attention=True)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_residual_with_cross_attention(self):
        """Test residual block with cross attention"""
        context_dim = 64
        block = ResidualBlock(
            self.channels, 
            use_attention=True,
            attention_type='cross',
            context_dim=context_dim
        )
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        context = torch.randn(self.batch_size, context_dim, self.height // 2, self.width // 2)
        
        output = block(x, context)
        
        self.assertEqual(output.shape, x.shape)


class TestAttentionModules(unittest.TestCase):
    """Test attention modules"""
    
    def setUp(self):
        self.batch_size = 2
        self.channels = 64
        self.height = 16
        self.width = 16
    
    def test_self_attention(self):
        """Test self attention module"""
        attention = SelfAttention(self.channels, num_heads=8)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        
        # Attention should preserve scale
        self.assertAlmostEqual(
            torch.std(output).item(),
            torch.std(x).item(),
            delta=1.0
        )
    
    def test_multi_scale_self_attention(self):
        """Test multi-scale self attention"""
        attention = MultiScaleSelfAttention(self.channels, scales=[1, 2])
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_cross_attention(self):
        """Test cross attention module"""
        context_dim = 48
        attention = CrossAttention(self.channels, context_dim, num_heads=4)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        context = torch.randn(self.batch_size, context_dim, self.height // 2, self.width // 2)
        
        output = attention(x, context)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_with_different_heads(self):
        """Test attention with different number of heads"""
        for num_heads in [1, 2, 4, 8]:
            with self.subTest(num_heads=num_heads):
                # Ensure channels is divisible by num_heads
                channels = 64
                attention = SelfAttention(channels, num_heads=num_heads)
                
                x = torch.randn(self.batch_size, channels, 8, 8)
                output = attention(x)
                
                self.assertEqual(output.shape, x.shape)


class TestLDPCEncoder(unittest.TestCase):
    """Test LDPC-aware encoder"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.encoder = LDPCAwareDualUNetEncoder(self.config)
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        cover_images = torch.randn(
            self.batch_size, 
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        # LDPC encoded messages (longer than original)
        max_encoded_length = int(self.config.message_length / (1 - self.config.ldpc_max_redundancy))
        ldpc_messages = torch.randint(0, 2, (self.batch_size, max_encoded_length), dtype=torch.float32)
        
        stego_images = self.encoder(cover_images, ldpc_messages)
        
        # Check output shape
        self.assertEqual(stego_images.shape, cover_images.shape)
        
        # Check output range (should be in [-1, 1] due to tanh)
        self.assertTrue(torch.all(stego_images >= -1))
        self.assertTrue(torch.all(stego_images <= 1))
    
    def test_encoder_with_different_message_lengths(self):
        """Test encoder with different message lengths"""
        cover_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        # Test with shorter and longer messages
        for factor in [0.5, 1.0, 1.5]:
            msg_length = int(self.config.message_length * factor)
            messages = torch.randint(0, 2, (self.batch_size, msg_length), dtype=torch.float32)
            
            stego_images = self.encoder(cover_images, messages)
            self.assertEqual(stego_images.shape, cover_images.shape)


class TestLDPCDecoder(unittest.TestCase):
    """Test LDPC-aware decoder"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.decoder = LDPCAwareDualUNetDecoder(self.config)
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        soft_codewords = self.decoder(stego_images)
        
        # Check output shape
        expected_length = int(self.config.message_length / (1 - self.config.ldpc_max_redundancy))
        self.assertEqual(soft_codewords.shape, (self.batch_size, expected_length))
        
        # Output should be soft values
        self.assertFalse(torch.all((soft_codewords == 0) | (soft_codewords == 1)))


class TestDualUNetDecoder(unittest.TestCase):
    """Test dual UNet decoder without LDPC"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.decoder = DualUNetDecoder(self.config)
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        messages = self.decoder(stego_images)
        
        # Check output shape
        self.assertEqual(messages.shape, (self.batch_size, self.config.message_length))
        
        # Output should be in tanh range
        self.assertTrue(torch.all(messages >= -1))
        self.assertTrue(torch.all(messages <= 1))


class TestRecoveryCVAE(unittest.TestCase):
    """Test recovery CVAE"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.cvae = RecoveryCVAE(self.config)
    
    def test_cvae_forward(self):
        """Test CVAE forward pass"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        cover_images = torch.randn_like(stego_images)
        
        recovered, mu, logvar = self.cvae(stego_images, cover_images)
        
        # Check output shapes
        self.assertEqual(recovered.shape, stego_images.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.config.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.config.latent_dim))
        
        # Recovered images should be in valid range
        self.assertTrue(torch.all(recovered >= -1))
        self.assertTrue(torch.all(recovered <= 1))
    
    def test_cvae_sampling(self):
        """Test CVAE sampling"""
        num_samples = 4
        condition = torch.randn(
            num_samples,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        samples = self.cvae.sample(num_samples, condition, torch.device('cpu'))
        
        self.assertEqual(samples.shape, condition.shape)
    
    def test_cvae_without_condition(self):
        """Test CVAE without conditioning"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        # Should use stego as condition if cover not provided
        recovered, mu, logvar = self.cvae(stego_images)
        
        self.assertEqual(recovered.shape, stego_images.shape)


class TestAdaptiveRecoveryCVAE(unittest.TestCase):
    """Test adaptive recovery CVAE"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 2
        self.adaptive_cvae = AdaptiveRecoveryCVAE(self.config)
    
    def test_adaptive_forward(self):
        """Test adaptive CVAE forward pass"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        recovered, mu, logvar = self.adaptive_cvae(stego_images)
        
        # Check outputs
        self.assertEqual(recovered.shape, stego_images.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.config.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.config.latent_dim))
    
    def test_severity_estimation(self):
        """Test attack severity estimation"""
        stego_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        # Get severity estimation
        severity_probs = self.adaptive_cvae.severity_estimator(stego_images)
        
        # Check output
        self.assertEqual(severity_probs.shape, (self.batch_size, 3))  # 3 severity levels
        
        # Should be probabilities
        self.assertTrue(torch.allclose(severity_probs.sum(dim=1), torch.ones(self.batch_size)))
        self.assertTrue(torch.all(severity_probs >= 0))
        self.assertTrue(torch.all(severity_probs <= 1))


class TestIntegration(unittest.TestCase):
    """Integration tests for model components"""
    
    def setUp(self):
        self.config = MockConfig()
        self.batch_size = 1
    
    def test_encoder_decoder_integration(self):
        """Test encoder-decoder integration"""
        encoder = LDPCAwareDualUNetEncoder(self.config)
        decoder = LDPCAwareDualUNetDecoder(self.config)
        
        # Create inputs
        cover_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size
        )
        
        max_encoded_length = int(self.config.message_length / (1 - self.config.ldpc_max_redundancy))
        ldpc_messages = torch.randint(0, 2, (self.batch_size, max_encoded_length), dtype=torch.float32)
        
        # Encode
        stego_images = encoder(cover_images, ldpc_messages)
        
        # Decode
        extracted_messages = decoder(stego_images)
        
        # Check compatibility
        self.assertEqual(extracted_messages.shape[0], ldpc_messages.shape[0])
        self.assertEqual(extracted_messages.shape[1], max_encoded_length)
    
    def test_gradient_flow(self):
        """Test gradient flow through models"""
        encoder = LDPCAwareDualUNetEncoder(self.config)
        decoder = LDPCAwareDualUNetDecoder(self.config)
        
        # Enable gradients
        encoder.train()
        decoder.train()
        
        # Create inputs
        cover_images = torch.randn(
            self.batch_size,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
            requires_grad=True
        )
        
        messages = torch.randint(0, 2, (self.batch_size, self.config.message_length), dtype=torch.float32)
        
        # Forward pass
        stego_images = encoder(cover_images, messages)
        extracted = decoder(stego_images)
        
        # Compute loss
        loss = nn.functional.mse_loss(extracted[:, :self.config.message_length], messages)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(cover_images.grad)
        
        # Check gradients flow through encoder
        for param in encoder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))
                break
        
        # Check gradients flow through decoder
        for param in decoder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))
                break


if __name__ == '__main__':
    unittest.main()