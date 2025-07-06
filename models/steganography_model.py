#!/usr/bin/env python3
"""
Complete LDPC Steganography Model
Integrates all components: dual UNet encoder/decoder + CVAE recovery + LDPC system
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

from .encoders.ldpc_aware_encoder import LDPCAwareDualUNetEncoder
from .decoders.ldpc_aware_decoder import LDPCAwareDualUNetDecoder
from .recovery.recovery_cvae import RecoveryCVAE
from .discriminator import Discriminator
from ..core.adaptive_ldpc import AdaptiveLDPC, create_ldpc_system
from ..training.attacks.attack_simulator import AttackSimulator


class AdvancedSteganographyModelWithLDPC(nn.Module):
    """Complete steganography model with LDPC error correction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main neural network components
        self.encoder = LDPCAwareDualUNetEncoder(config)
        self.decoder = LDPCAwareDualUNetDecoder(config)
        self.recovery_net = RecoveryCVAE(config)
        self.discriminator = Discriminator(config)
        
        # LDPC system for error correction
        self.ldpc_system = create_ldpc_system(config)
        
        # Attack simulator for training
        self.attack_simulator = AttackSimulator(config.device)
        
        # Model state
        self.training_phase = "initialization"
        
        # Log model initialization
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model initialization information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"🔧 LDPC Steganography Model initialized:")
        logging.info(f"   Original message length: {self.config.message_length}")
        logging.info(f"   Max encoded length: {self.encoder.max_encoded_length}")
        logging.info(f"   LDPC redundancy range: {self.config.ldpc_min_redundancy:.1f} - {self.config.ldpc_max_redundancy:.1f}")
        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    def forward(self, cover_images: torch.Tensor, messages: torch.Tensor, 
                attack_type: str = 'none', attack_strength: float = 0.5, 
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with LDPC
        
        Args:
            cover_images: Cover images [batch_size, channels, height, width]
            messages: Binary messages [batch_size, message_length]
            attack_type: Type of attack to simulate
            attack_strength: Strength of attack (0-1)
            training: Whether in training mode
            
        Returns:
            Dictionary containing all outputs
        """
        
        batch_size = cover_images.size(0)
        
        # Step 1: Encode messages with adaptive LDPC
        try:
            ldpc_encoded_messages = self.ldpc_system.encode(messages, attack_strength)
        except Exception as e:
            logging.warning(f"LDPC encoding failed: {e}, using original messages")
            ldpc_encoded_messages = messages
        
        # Step 2: Generate stego images using dual UNet encoder
        stego_images = self.encoder(cover_images, ldpc_encoded_messages)
        
        # Step 3: Apply attacks during training
        if training and attack_type != 'none':
            attacked_images = self.attack_simulator.apply_attack(
                stego_images, attack_type, attack_strength
            )
        else:
            attacked_images = stego_images
        
        # Step 4: Extract messages using dual UNet decoder (soft output)
        decoded_ldpc_soft = self.decoder(attacked_images)
        
        # Step 5: LDPC error correction to get original messages
        try:
            decoded_messages = self.ldpc_system.decode(
                decoded_ldpc_soft, attack_strength, use_soft=True
            )
        except Exception as e:
            logging.warning(f"LDPC decoding failed: {e}, using threshold decoding")
            decoded_messages = torch.sigmoid(decoded_ldpc_soft)
        
        # Step 6: Image recovery using CVAE
        recovered_images, mu, logvar = self.recovery_net(attacked_images, cover_images)
        
        # Step 7: Discriminator prediction (for adversarial training)
        discriminator_pred = None
        if training:
            discriminator_pred = self.discriminator(stego_images)
        
        # Return comprehensive outputs
        return {
            'stego_images': stego_images,
            'attacked_images': attacked_images,
            'ldpc_encoded_messages': ldpc_encoded_messages,
            'decoded_ldpc_soft': decoded_ldpc_soft,
            'decoded_messages': decoded_messages,
            'recovered_images': recovered_images,
            'mu': mu,
            'logvar': logvar,
            'discriminator_pred': discriminator_pred
        }
    
    def encode_message(self, cover_image: torch.Tensor, message: torch.Tensor, 
                      attack_strength: float = 0.3) -> torch.Tensor:
        """
        Encode a single message into cover image
        
        Args:
            cover_image: Single cover image [1, channels, height, width]
            message: Binary message [1, message_length]
            attack_strength: Expected attack strength
            
        Returns:
            Stego image [1, channels, height, width]
        """
        
        self.eval()
        with torch.no_grad():
            # LDPC encode
            ldpc_encoded = self.ldpc_system.encode(message, attack_strength)
            
            # Generate stego image
            stego_image = self.encoder(cover_image, ldpc_encoded)
            
        return stego_image
    
    def decode_message(self, stego_image: torch.Tensor, 
                      attack_strength: float = 0.3) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode message from stego image
        
        Args:
            stego_image: Stego image [1, channels, height, width]
            attack_strength: Attack strength used during encoding
            
        Returns:
            Tuple of (decoded_message, decoding_info)
        """
        
        self.eval()
        with torch.no_grad():
            # Extract soft LDPC codeword
            decoded_soft = self.decoder(stego_image)
            
            # LDPC decode
            try:
                decoded_message = self.ldpc_system.decode(decoded_soft, attack_strength, use_soft=True)
                decoding_info = {'ldpc_decoding': 'success'}
            except Exception as e:
                # Fallback to threshold decoding
                decoded_message = (torch.sigmoid(decoded_soft) > 0.5).float()
                decoding_info = {'ldpc_decoding': 'failed', 'error': str(e)}
        
        return decoded_message, decoding_info
    
    def recover_image(self, stego_image: torch.Tensor, 
                     cover_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Recover original image from stego image
        
        Args:
            stego_image: Stego image [1, channels, height, width]
            cover_image: Original cover image (optional, for conditioning)
            
        Returns:
            Recovered image [1, channels, height, width]
        """
        
        self.eval()
        with torch.no_grad():
            if cover_image is None:
                # Use stego image as condition (self-recovery)
                recovered_image, _, _ = self.recovery_net(stego_image, stego_image)
            else:
                recovered_image, _, _ = self.recovery_net(stego_image, cover_image)
        
        return recovered_image
    
    def get_ldpc_info(self, attack_strength: float = 0.3) -> Dict[str, Any]:
        """Get LDPC code information for given attack strength"""
        return self.ldpc_system.get_code_info(attack_strength)
    
    def set_training_phase(self, phase: str):
        """Set current training phase for adaptive behavior"""
        self.training_phase = phase
        logging.info(f"Training phase set to: {phase}")
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state information"""
        return {
            'training_phase': self.training_phase,
            'ldpc_system_info': {
                'message_length': self.ldpc_system.message_length,
                'redundancy_levels': self.ldpc_system.redundancy_levels,
                'neural_decoder_available': self.ldpc_system.neural_decoder is not None
            },
            'model_info': {
                'encoder_max_length': self.encoder.max_encoded_length,
                'decoder_max_length': self.decoder.max_encoded_length,
                'device': next(self.parameters()).device
            }
        }
    
    def validate_model_integrity(self) -> Dict[str, bool]:
        """Validate model components integrity"""
        
        validation_results = {}
        
        # Test LDPC system
        try:
            test_message = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
            test_encoded = self.ldpc_system.encode(test_message, 0.3)
            test_decoded = self.ldpc_system.decode(test_encoded, 0.3)
            validation_results['ldpc_system'] = True
        except Exception as e:
            logging.error(f"LDPC system validation failed: {e}")
            validation_results['ldpc_system'] = False
        
        # Test encoder/decoder compatibility
        try:
            test_cover = torch.randn(1, self.config.channels, 
                                   self.config.image_size, self.config.image_size)
            test_message = torch.randint(0, 2, (1, self.config.message_length), dtype=torch.float32)
            
            # Forward pass
            outputs = self.forward(test_cover, test_message, training=False)
            
            # Check output shapes
            expected_shapes = {
                'stego_images': test_cover.shape,
                'decoded_messages': test_message.shape,
                'recovered_images': test_cover.shape
            }
            
            shapes_correct = all(
                outputs[key].shape == expected_shapes[key] 
                for key in expected_shapes if key in outputs
            )
            
            validation_results['model_forward'] = shapes_correct
            
        except Exception as e:
            logging.error(f"Model forward validation failed: {e}")
            validation_results['model_forward'] = False
        
        # Test recovery network
        try:
            test_stego = torch.randn_like(test_cover)
            recovered, mu, logvar = self.recovery_net(test_stego, test_cover)
            validation_results['recovery_net'] = recovered.shape == test_cover.shape
        except Exception as e:
            logging.error(f"Recovery network validation failed: {e}")
            validation_results['recovery_net'] = False
        
        # Test discriminator
        try:
            disc_output = self.discriminator(test_cover)
            validation_results['discriminator'] = disc_output.numel() == test_cover.size(0)
        except Exception as e:
            logging.error(f"Discriminator validation failed: {e}")
            validation_results['discriminator'] = False
        
        return validation_results
    
    def calculate_model_capacity(self) -> Dict[str, float]:
        """Calculate effective model capacity"""
        
        capacity_info = {}
        
        # Base capacity (without ECC)
        base_capacity = self.config.message_length / (
            self.config.image_size * self.config.image_size * self.config.channels
        )
        capacity_info['base_capacity_bpp'] = base_capacity
        
        # Effective capacity for different attack strengths
        attack_strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        for attack_strength in attack_strengths:
            ldpc_info = self.get_ldpc_info(attack_strength)
            effective_capacity = (ldpc_info['k'] / ldpc_info['expansion_factor']) / (
                self.config.image_size * self.config.image_size * self.config.channels
            )
            capacity_info[f'effective_capacity_attack_{attack_strength}'] = effective_capacity
        
        return capacity_info
    
    def export_model_summary(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive model summary"""
        
        summary = {
            'model_type': 'LDPC_Steganography_Model',
            'configuration': {
                'message_length': self.config.message_length,
                'image_size': self.config.image_size,
                'channels': self.config.channels,
                'ldpc_redundancy_range': (self.config.ldpc_min_redundancy, 
                                         self.config.ldpc_max_redundancy)
            },
            'model_state': self.get_model_state(),
            'validation_results': self.validate_model_integrity(),
            'capacity_analysis': self.calculate_model_capacity(),
            'component_info': {
                'encoder': type(self.encoder).__name__,
                'decoder': type(self.decoder).__name__,
                'recovery_net': type(self.recovery_net).__name__,
                'discriminator': type(self.discriminator).__name__,
                'ldpc_system': type(self.ldpc_system).__name__
            }
        }
        
        # Save summary if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logging.info(f"Model summary saved to {save_path}")
        
        return summary