#!/usr/bin/env python3
"""
Adaptive LDPC System
Complete adaptive LDPC system with attack-strength dependent redundancy
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Any
import logging
from .ldpc_generator import LDPCGenerator
from .ldpc_encoder import LDPCEncoder, ParallelLDPCEncoder
from .ldpc_decoder import LDPCDecoder, NeuralLDPCDecoder


class AdaptiveLDPC:
    """Adaptive LDPC system with attack-strength dependent redundancy"""
    
    def __init__(self, 
                 message_length: int,
                 min_redundancy: float = 0.1,
                 max_redundancy: float = 0.5,
                 device: str = 'cpu',
                 use_neural_decoder: bool = True,
                 use_parallel_encoder: bool = False):
        """
        Initialize adaptive LDPC system
        
        Args:
            message_length: Original message length in bits
            min_redundancy: Minimum redundancy ratio
            max_redundancy: Maximum redundancy ratio
            device: PyTorch device
            use_neural_decoder: Whether to use neural decoder
            use_parallel_encoder: Whether to use parallel encoder
        """
        self.message_length = message_length
        self.min_redundancy = min_redundancy
        self.max_redundancy = max_redundancy
        self.device = device
        self.use_neural_decoder = use_neural_decoder
        self.use_parallel_encoder = use_parallel_encoder
        
        # Pre-generate LDPC codes for different redundancy levels
        self.ldpc_codes = {}
        self.encoders = {}
        self.decoders = {}
        
        # Define redundancy levels
        self.redundancy_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        # Filter levels within specified range
        self.redundancy_levels = [r for r in self.redundancy_levels 
                                if min_redundancy <= r <= max_redundancy]
        
        if not self.redundancy_levels:
            self.redundancy_levels = [min_redundancy, max_redundancy]
        
        self._generate_ldpc_codes()
        
        # Neural decoder (if enabled)
        self.neural_decoder = None
        if use_neural_decoder and device != 'cpu':
            self._create_neural_decoder()
        
        logging.info(f"Adaptive LDPC initialized: {len(self.redundancy_levels)} redundancy levels")
    
    def _generate_ldpc_codes(self):
        """Pre-generate LDPC codes for different redundancy levels"""
        
        logging.info("Generating LDPC codes for different redundancy levels...")
        
        for redundancy in self.redundancy_levels:
            k = self.message_length  # Information bits
            n = int(k / (1 - redundancy))  # Total bits after encoding
            
            # Ensure n is reasonable
            if n > k * 3:  # Limit maximum expansion
                n = k * 3
                redundancy = 1 - k/n  # Recalculate actual redundancy
            
            try:
                # Generate LDPC code
                ldpc_gen = LDPCGenerator(n=n, k=k, seed=42, construction="gallager")
                
                # Validate the generated code
                if not ldpc_gen.validate_code():
                    logging.warning(f"Invalid LDPC code for redundancy {redundancy:.2f}")
                    continue
                
                # Create encoder
                if self.use_parallel_encoder:
                    encoder = ParallelLDPCEncoder(ldpc_gen, method="systematic")
                else:
                    encoder = LDPCEncoder(ldpc_gen, method="systematic")
                
                # Create decoder
                decoder = LDPCDecoder(ldpc_gen, method="belief_propagation", max_iterations=50)
                
                # Store components
                self.ldpc_codes[redundancy] = ldpc_gen
                self.encoders[redundancy] = encoder
                self.decoders[redundancy] = decoder
                
                logging.info(f"   ✅ Redundancy {redundancy:.2f}: k={k}, n={n}, rate={k/n:.3f}")
                
            except Exception as e:
                logging.error(f"   ❌ Failed to generate LDPC for redundancy {redundancy}: {e}")
        
        if not self.ldpc_codes:
            raise RuntimeError("No valid LDPC codes generated")
    
    def _create_neural_decoder(self):
        """Create neural network for soft LDPC decoding"""
        
        try:
            # Find maximum encoded length
            max_n = max([ldpc.n for ldpc in self.ldpc_codes.values()])
            
            # Use a representative LDPC code for neural decoder
            representative_redundancy = self.redundancy_levels[len(self.redundancy_levels)//2]
            representative_ldpc = self.ldpc_codes[representative_redundancy]
            
            self.neural_decoder = NeuralLDPCDecoder(
                representative_ldpc,
                hidden_dims=[max_n * 2, max_n * 2, max_n],
                dropout_rate=0.1
            ).to(self.device)
            
            logging.info(f"🧠 Created neural LDPC decoder for max length: {max_n}")
            
        except Exception as e:
            logging.warning(f"Neural decoder creation failed: {e}")
            self.neural_decoder = None
    
    def calculate_adaptive_redundancy(self, attack_strength: float) -> float:
        """Calculate optimal redundancy based on attack strength"""
        
        # Exponential scaling for attack strength
        redundancy = self.min_redundancy + \
                    (self.max_redundancy - self.min_redundancy) * (attack_strength ** 0.7)
        
        # Snap to available levels
        closest_level = min(self.redundancy_levels, key=lambda x: abs(x - redundancy))
        
        return closest_level
    
    def encode(self, messages: torch.Tensor, attack_strength: float = 0.3) -> torch.Tensor:
        """
        Encode messages with adaptive LDPC
        
        Args:
            messages: Input messages [batch_size, message_length]
            attack_strength: Expected attack strength (0-1)
            
        Returns:
            encoded_messages: LDPC encoded messages [batch_size, encoded_length]
        """
        
        batch_size = messages.size(0)
        
        # Calculate optimal redundancy
        redundancy = self.calculate_adaptive_redundancy(attack_strength)
        
        if redundancy not in self.ldpc_codes:
            logging.warning(f"Redundancy {redundancy} not available, using closest")
            redundancy = min(self.redundancy_levels, key=lambda x: abs(x - redundancy))
        
        encoder = self.encoders[redundancy]
        
        # Convert to numpy for encoding
        messages_np = messages.cpu().numpy()
        
        # Encode batch
        if self.use_parallel_encoder and hasattr(encoder, 'encode_batch_parallel'):
            encoded_batch = encoder.encode_batch_parallel(messages_np)
        else:
            encoded_batch = encoder.encode_batch(messages_np)
        
        # Convert back to tensor
        encoded_tensor = torch.tensor(encoded_batch, dtype=torch.float32, device=messages.device)
        
        return encoded_tensor
    
    def decode(self, received_messages: torch.Tensor, attack_strength: float = 0.3, 
              use_soft: bool = True, noise_variance: float = 1.0) -> torch.Tensor:
        """
        Decode messages with LDPC error correction
        
        Args:
            received_messages: Received (possibly corrupted) messages
            attack_strength: Attack strength used during encoding
            use_soft: Use soft decoding if available
            noise_variance: Channel noise variance
            
        Returns:
            decoded_messages: Corrected original messages
        """
        
        batch_size = received_messages.size(0)
        
        # Get the LDPC code used for encoding
        redundancy = self.calculate_adaptive_redundancy(attack_strength)
        if redundancy not in self.ldpc_codes:
            redundancy = min(self.redundancy_levels, key=lambda x: abs(x - redundancy))
        
        ldpc_code = self.ldpc_codes[redundancy]
        decoder = self.decoders[redundancy]
        
        # Try neural decoding first if available
        if use_soft and self.neural_decoder is not None:
            try:
                decoded_bits, neural_info = self._neural_decode(received_messages, ldpc_code)
                
                # If neural decoding successful for most samples, use it
                if neural_info.get('neural_valid_rate', 0) > 0.8:
                    return self._extract_information_bits(decoded_bits, ldpc_code)
                else:
                    logging.info("Neural decoding quality low, falling back to traditional")
            except Exception as e:
                logging.warning(f"Neural decoding failed: {e}, using traditional")
        
        # Traditional LDPC decoding
        return self._traditional_decode(received_messages, decoder, ldpc_code, noise_variance)
    
    def _neural_decode(self, received_messages: torch.Tensor, 
                      ldpc_code: LDPCGenerator) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Neural network decoding"""
        
        # Pad/truncate to neural decoder expected length
        target_length = self.neural_decoder.n
        current_length = received_messages.size(1)
        
        if current_length < target_length:
            padding = torch.zeros(received_messages.size(0), target_length - current_length,
                                device=received_messages.device)
            padded_messages = torch.cat([received_messages, padding], dim=1)
        else:
            padded_messages = received_messages[:, :target_length]
        
        # Neural decoding
        decoded_bits, neural_info = self.neural_decoder.decode(padded_messages)
        
        # Truncate to actual code length
        if decoded_bits.size(1) > ldpc_code.n:
            decoded_bits = decoded_bits[:, :ldpc_code.n]
        
        return decoded_bits, neural_info
    
    def _traditional_decode(self, received_messages: torch.Tensor, 
                           decoder: LDPCDecoder, ldpc_code: LDPCGenerator,
                           noise_variance: float) -> torch.Tensor:
        """Traditional LDPC decoding"""
        
        # Convert to numpy
        received_np = received_messages.cpu().numpy()
        
        # Decode batch
        decoded_batch, decoding_info_list = decoder.decode_batch(received_np, noise_variance)
        
        # Extract information bits
        info_batch = []
        for i in range(decoded_batch.shape[0]):
            info_bits = decoder.extract_information_bits(decoded_batch[i])
            info_batch.append(info_bits)
        
        # Pad to original message length
        info_batch = self._pad_to_message_length(np.array(info_batch))
        
        # Convert back to tensor
        return torch.tensor(info_batch, dtype=torch.float32, device=received_messages.device)
    
    def _extract_information_bits(self, decoded_bits: torch.Tensor, 
                                 ldpc_code: LDPCGenerator) -> torch.Tensor:
        """Extract information bits from decoded codewords"""
        
        # For systematic codes, information bits are the first k bits
        info_bits = decoded_bits[:, :ldpc_code.k]
        
        # Pad to original message length
        info_bits_np = info_bits.cpu().numpy()
        padded_info = self._pad_to_message_length(info_bits_np)
        
        return torch.tensor(padded_info, dtype=torch.float32, device=decoded_bits.device)
    
    def _pad_to_message_length(self, info_batch: np.ndarray) -> np.ndarray:
        """Pad information bits to original message length"""
        
        batch_size = info_batch.shape[0]
        current_length = info_batch.shape[1]
        
        if current_length > self.message_length:
            return info_batch[:, :self.message_length]
        elif current_length < self.message_length:
            padding = np.zeros((batch_size, self.message_length - current_length))
            return np.concatenate([info_batch, padding], axis=1)
        else:
            return info_batch
    
    def get_code_info(self, attack_strength: float = 0.3) -> Dict[str, Any]:
        """Get information about the LDPC code for given attack strength"""
        
        redundancy = self.calculate_adaptive_redundancy(attack_strength)
        
        if redundancy not in self.ldpc_codes:
            return {'error': 'LDPC code not available'}
        
        ldpc_code = self.ldpc_codes[redundancy]
        
        return {
            'redundancy': redundancy,
            'k': ldpc_code.k,
            'n': ldpc_code.n,
            'rate': ldpc_code.rate,
            'parity_bits': ldpc_code.r,
            'expansion_factor': ldpc_code.n / ldpc_code.k,
            'available_redundancies': self.redundancy_levels,
            'neural_decoder_available': self.neural_decoder is not None
        }
    
    def benchmark_performance(self, num_tests: int = 1000, 
                            noise_levels: List[float] = [0.1, 0.2, 0.3]) -> Dict[str, Any]:
        """Benchmark LDPC performance across different configurations"""
        
        results = {}
        
        for redundancy in self.redundancy_levels:
            encoder = self.encoders[redundancy]
            decoder = self.decoders[redundancy]
            ldpc_code = self.ldpc_codes[redundancy]
            
            redundancy_results = {}
            
            for noise_level in noise_levels:
                # Generate random test messages
                test_messages = torch.randint(0, 2, (num_tests, self.message_length), dtype=torch.float32)
                
                # Encode
                encoded = encoder.encode_batch(test_messages.numpy())
                
                # Add noise
                noise = np.random.normal(0, noise_level, encoded.shape)
                noisy_encoded = encoded + noise
                
                # Decode
                decoded_batch, _ = decoder.decode_batch(noisy_encoded, noise_level**2)
                
                # Extract information bits
                info_batch = []
                for i in range(decoded_batch.shape[0]):
                    info_bits = decoder.extract_information_bits(decoded_batch[i])
                    info_batch.append(info_bits)
                
                info_batch = self._pad_to_message_length(np.array(info_batch))
                
                # Calculate BER
                original_bits = (test_messages > 0.5).numpy()
                decoded_bits = (info_batch > 0.5)
                
                errors = np.sum(np.abs(original_bits - decoded_bits))
                total_bits = original_bits.size
                ber = errors / total_bits
                
                redundancy_results[noise_level] = {
                    'ber': ber,
                    'total_errors': int(errors),
                    'total_bits': int(total_bits)
                }
            
            results[redundancy] = redundancy_results
        
        return results
    
    def train_neural_decoder(self, train_data: torch.Tensor, epochs: int = 100):
        """Train neural decoder on synthetic data"""
        
        if self.neural_decoder is None:
            logging.warning("Neural decoder not available")
            return
        
        logging.info(f"Training neural LDPC decoder for {epochs} epochs...")
        
        # Create training dataset
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate synthetic training pairs
        num_samples = len(train_data) * 10  # Augment data
        synthetic_messages = torch.randint(0, 2, (num_samples, self.message_length), dtype=torch.float32)
        
        # Use medium redundancy for training
        training_redundancy = self.redundancy_levels[len(self.redundancy_levels)//2]
        encoder = self.encoders[training_redundancy]
        
        # Encode messages
        encoded_messages = encoder.encode_batch(synthetic_messages.numpy())
        encoded_tensor = torch.tensor(encoded_messages, dtype=torch.float32)
        
        # Add various noise levels
        training_pairs = []
        for noise_level in [0.05, 0.1, 0.15, 0.2]:
            noise = torch.randn_like(encoded_tensor) * noise_level
            noisy_encoded = encoded_tensor + noise
            training_pairs.append((noisy_encoded, encoded_tensor))
        
        # Combine all training pairs
        all_noisy = torch.cat([pair[0] for pair in training_pairs], dim=0)
        all_clean = torch.cat([pair[1] for pair in training_pairs], dim=0)
        
        # Create data loader
        dataset = TensorDataset(all_noisy, all_clean)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train the neural decoder
        self.neural_decoder.train_decoder(train_loader, epochs, device=self.device)
        
        logging.info("✅ Neural decoder training completed")


class LDPCPerformanceAnalyzer:
    """Performance analyzer for LDPC systems"""
    
    def __init__(self, adaptive_ldpc: AdaptiveLDPC):
        """
        Initialize performance analyzer
        
        Args:
            adaptive_ldpc: AdaptiveLDPC system to analyze
        """
        self.ldpc_system = adaptive_ldpc
        
    def compare_with_reed_solomon(self, num_tests: int = 1000) -> Dict[str, Any]:
        """Compare LDPC performance with simulated Reed-Solomon"""
        
        logging.info("Comparing LDPC with Reed-Solomon (simulated)...")
        
        message_length = self.ldpc_system.message_length
        noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
        attack_strength = 0.3
        
        ldpc_results = []
        rs_results = []  # Simulated Reed-Solomon results
        
        for noise_level in noise_levels:
            # LDPC test
            try:
                test_messages = torch.randint(0, 2, (num_tests, message_length), dtype=torch.float32)
                
                # LDPC encode
                encoded_ldpc = self.ldpc_system.encode(test_messages, attack_strength)
                
                # Add noise
                noise = torch.randn_like(encoded_ldpc) * noise_level
                noisy_ldpc = encoded_ldpc + noise
                
                # LDPC decode
                decoded_ldpc = self.ldpc_system.decode(noisy_ldpc, attack_strength)
                
                # Calculate LDPC BER
                original_bits = (test_messages > 0.5).float()
                decoded_bits = (decoded_ldpc > 0.5).float()
                ldpc_ber = torch.sum(torch.abs(original_bits - decoded_bits)).item() / original_bits.numel()
                
                ldpc_results.append(ldpc_ber)
                
            except Exception as e:
                logging.error(f"LDPC test failed at noise {noise_level}: {e}")
                ldpc_results.append(1.0)
            
            # Simulated Reed-Solomon (simple repetition code simulation)
            repetition_factor = 3  # 3x repetition for fairness
            repeated_msg = test_messages.repeat(1, repetition_factor)
            noise_rs = torch.randn_like(repeated_msg) * noise_level
            noisy_rs = repeated_msg + noise_rs
            
            # Majority voting decode
            decoded_rs = noisy_rs.view(num_tests, message_length, repetition_factor)
            decoded_rs = (torch.mean(decoded_rs, dim=2) > 0.5).float()
            
            rs_ber = torch.sum(torch.abs(original_bits - decoded_rs)).item() / original_bits.numel()
            rs_results.append(rs_ber)
        
        # Create comparison report
        comparison = {
            'noise_levels': noise_levels,
            'ldpc_ber': ldpc_results,
            'reed_solomon_ber': rs_results,
            'improvements': []
        }
        
        for i, noise in enumerate(noise_levels):
            if rs_results[i] > 0:
                improvement = ((rs_results[i] - ldpc_results[i]) / rs_results[i] * 100)
                comparison['improvements'].append(improvement)
            else:
                comparison['improvements'].append(0)
        
        return comparison
    
    def analyze_redundancy_efficiency(self) -> Dict[str, Any]:
        """Analyze efficiency of different redundancy levels"""
        
        efficiency_analysis = {}
        
        for redundancy in self.ldpc_system.redundancy_levels:
            code_info = self.ldpc_system.ldpc_codes[redundancy]
            
            efficiency_analysis[redundancy] = {
                'code_rate': code_info.rate,
                'expansion_factor': code_info.n / code_info.k,
                'redundancy_bits': code_info.r,
                'efficiency_score': code_info.rate / (1 + redundancy),  # Simple efficiency metric
                'properties': code_info.get_properties()
            }
        
        return efficiency_analysis
    
    def capacity_analysis(self, attack_strengths: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) -> Dict[str, Any]:
        """Analyze capacity vs robustness trade-off"""
        
        capacity_data = {
            'attack_strengths': attack_strengths,
            'redundancies': [],
            'code_rates': [],
            'expansion_factors': [],
            'effective_capacities': []
        }
        
        for attack_strength in attack_strengths:
            redundancy = self.ldpc_system.calculate_adaptive_redundancy(attack_strength)
            code_info = self.ldpc_system.get_code_info(attack_strength)
            
            capacity_data['redundancies'].append(redundancy)
            capacity_data['code_rates'].append(code_info['rate'])
            capacity_data['expansion_factors'].append(code_info['expansion_factor'])
            
            # Effective capacity considering overhead
            effective_capacity = code_info['rate'] / code_info['expansion_factor']
            capacity_data['effective_capacities'].append(effective_capacity)
        
        return capacity_data
    
    def create_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive performance report"""
        
        report = {
            'system_info': {
                'message_length': self.ldpc_system.message_length,
                'redundancy_range': (self.ldpc_system.min_redundancy, self.ldpc_system.max_redundancy),
                'available_levels': self.ldpc_system.redundancy_levels,
                'neural_decoder': self.ldpc_system.neural_decoder is not None,
                'parallel_encoder': self.ldpc_system.use_parallel_encoder
            },
            'ldpc_vs_reed_solomon': self.compare_with_reed_solomon(),
            'redundancy_efficiency': self.analyze_redundancy_efficiency(),
            'capacity_analysis': self.capacity_analysis(),
            'code_properties': {}
        }
        
        # Add individual code properties
        for redundancy in self.ldpc_system.redundancy_levels:
            ldpc_code = self.ldpc_system.ldpc_codes[redundancy]
            report['code_properties'][redundancy] = ldpc_code.get_properties()
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Performance report saved to {save_path}")
        
        return report


def create_ldpc_system(config) -> AdaptiveLDPC:
    """
    Factory function to create LDPC system from configuration
    
    Args:
        config: Configuration object with LDPC settings
        
    Returns:
        Configured AdaptiveLDPC system
    """
    
    ldpc_system = AdaptiveLDPC(
        message_length=config.message_length,
        min_redundancy=getattr(config, 'ldpc_min_redundancy', 0.1),
        max_redundancy=getattr(config, 'ldpc_max_redundancy', 0.5),
        device=config.device,
        use_neural_decoder=getattr(config, 'ldpc_use_neural_decoder', True),
        use_parallel_encoder=getattr(config, 'ldpc_parallel_encoder', False)
    )
    
    # Train neural decoder if requested
    if (hasattr(config, 'ldpc_train_neural_decoder') and 
        config.ldpc_train_neural_decoder and 
        ldpc_system.neural_decoder is not None):
        
        # Generate synthetic training data
        synthetic_messages = torch.randint(0, 2, (1000, config.message_length), dtype=torch.float32)
        ldpc_system.train_neural_decoder(synthetic_messages, 
                                       epochs=getattr(config, 'ldpc_neural_epochs', 100))
    
    return ldpc_system


# Export main classes
__all__ = [
    'AdaptiveLDPC',
    'LDPCPerformanceAnalyzer', 
    'create_ldpc_system'
]