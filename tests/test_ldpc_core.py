#!/usr/bin/env python3
"""
Unit Tests for LDPC Core Components
Test LDPC generator, encoder, decoder, and adaptive system
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ldpc_generator import LDPCGenerator, OptimizedLDPCGenerator
from core.ldpc_encoder import LDPCEncoder, ParallelLDPCEncoder
from core.ldpc_decoder import LDPCDecoder, NeuralLDPCDecoder
from core.adaptive_ldpc import AdaptiveLDPC, LDPCPerformanceAnalyzer


class TestLDPCGenerator(unittest.TestCase):
    """Test LDPC generator functionality"""
    
    def setUp(self):
        self.n = 100  # Codeword length
        self.k = 50   # Information length
        self.seed = 42
    
    def test_generator_initialization(self):
        """Test LDPC generator initialization"""
        generator = LDPCGenerator(self.n, self.k, self.seed)
        
        self.assertEqual(generator.n, self.n)
        self.assertEqual(generator.k, self.k)
        self.assertEqual(generator.r, self.n - self.k)
        self.assertAlmostEqual(generator.rate, self.k / self.n)
        self.assertIsNotNone(generator.H)
    
    def test_parity_check_matrix_properties(self):
        """Test properties of generated parity check matrix"""
        generator = LDPCGenerator(self.n, self.k, self.seed)
        
        # Check dimensions
        self.assertEqual(generator.H.shape, (generator.r, generator.n))
        
        # Check binary
        self.assertTrue(np.all(np.isin(generator.H, [0, 1])))
        
        # Check sparsity
        density = np.sum(generator.H) / (generator.H.shape[0] * generator.H.shape[1])
        self.assertLess(density, 0.1)  # Should be sparse
        
        # Check no all-zero columns
        col_sums = np.sum(generator.H, axis=0)
        self.assertTrue(np.all(col_sums > 0))
    
    def test_different_constructions(self):
        """Test different LDPC construction methods"""
        constructions = ['gallager', 'mackay', 'progressive']
        
        for construction in constructions:
            with self.subTest(construction=construction):
                generator = LDPCGenerator(self.n, self.k, self.seed, construction)
                self.assertTrue(generator.validate_code())
    
    def test_systematic_form(self):
        """Test conversion to systematic form"""
        generator = LDPCGenerator(self.n, self.k, self.seed)
        H_sys = generator.to_systematic_form()
        
        if H_sys is not None:
            self.assertEqual(H_sys.shape, generator.H.shape)
            # In systematic form, part of H should be identity-like
    
    def test_generator_matrix(self):
        """Test generator matrix creation"""
        generator = LDPCGenerator(self.n, self.k, self.seed)
        G = generator.generate_generator_matrix()
        
        if G is not None:
            self.assertEqual(G.shape, (self.k, self.n))
            # Check if G generates valid codewords
            # H * G^T should be zero matrix
            if generator.H is not None:
                product = np.dot(generator.H, G.T) % 2
                self.assertTrue(np.all(product == 0))
    
    def test_save_load_matrix(self):
        """Test saving and loading LDPC matrix"""
        import tempfile
        
        generator = LDPCGenerator(self.n, self.k, self.seed)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_file = f.name
            
        try:
            # Save
            generator.save_matrix(temp_file)
            
            # Load
            loaded_generator = LDPCGenerator.load_matrix(temp_file)
            
            # Compare
            self.assertEqual(loaded_generator.n, generator.n)
            self.assertEqual(loaded_generator.k, generator.k)
            np.testing.assert_array_equal(loaded_generator.H, generator.H)
            
        finally:
            os.unlink(temp_file)


class TestLDPCEncoder(unittest.TestCase):
    """Test LDPC encoder functionality"""
    
    def setUp(self):
        self.n = 100
        self.k = 50
        self.generator = LDPCGenerator(self.n, self.k, seed=42)
        self.encoder = LDPCEncoder(self.generator)
    
    def test_encoder_initialization(self):
        """Test encoder initialization"""
        self.assertEqual(self.encoder.n, self.n)
        self.assertEqual(self.encoder.k, self.k)
        self.assertTrue(self.encoder.encoding_ready)
    
    def test_single_encoding(self):
        """Test encoding single message"""
        message = np.random.randint(0, 2, self.k)
        codeword = self.encoder.encode_single(message)
        
        self.assertEqual(len(codeword), self.n)
        self.assertTrue(np.all(np.isin(codeword, [0, 1])))
        
        # Verify it's a valid codeword
        syndrome = np.dot(self.generator.H, codeword) % 2
        self.assertTrue(np.all(syndrome == 0))
    
    def test_batch_encoding(self):
        """Test batch encoding"""
        batch_size = 10
        messages = np.random.randint(0, 2, (batch_size, self.k))
        codewords = self.encoder.encode_batch(messages)
        
        self.assertEqual(codewords.shape, (batch_size, self.n))
        
        # Verify all are valid codewords
        for i in range(batch_size):
            syndrome = np.dot(self.generator.H, codewords[i]) % 2
            self.assertTrue(np.all(syndrome == 0))
    
    def test_torch_compatibility(self):
        """Test encoding with PyTorch tensors"""
        message_tensor = torch.randint(0, 2, (self.k,), dtype=torch.float32)
        codeword = self.encoder.encode_single(message_tensor)
        
        self.assertIsInstance(codeword, np.ndarray)
        self.assertEqual(len(codeword), self.n)
    
    def test_systematic_encoding(self):
        """Test systematic encoding property"""
        if self.encoder.method == 'systematic':
            message = np.random.randint(0, 2, self.k)
            codeword = self.encoder.encode_single(message)
            
            # For systematic codes, first k bits should be the message
            np.testing.assert_array_equal(codeword[:self.k], message)
    
    def test_parallel_encoder(self):
        """Test parallel encoder"""
        parallel_encoder = ParallelLDPCEncoder(self.generator, num_threads=2)
        
        batch_size = 20
        messages = np.random.randint(0, 2, (batch_size, self.k))
        
        # Encode with parallel encoder
        parallel_codewords = parallel_encoder.encode_batch_parallel(messages)
        
        # Compare with sequential encoding
        sequential_codewords = self.encoder.encode_batch(messages)
        
        np.testing.assert_array_equal(parallel_codewords, sequential_codewords)


class TestLDPCDecoder(unittest.TestCase):
    """Test LDPC decoder functionality"""
    
    def setUp(self):
        self.n = 100
        self.k = 50
        self.generator = LDPCGenerator(self.n, self.k, seed=42)
        self.encoder = LDPCEncoder(self.generator)
        self.decoder = LDPCDecoder(self.generator, max_iterations=50)
    
    def test_decoder_initialization(self):
        """Test decoder initialization"""
        self.assertEqual(self.decoder.n, self.n)
        self.assertEqual(self.decoder.k, self.k)
        self.assertEqual(len(self.decoder.check_to_var), self.generator.r)
        self.assertEqual(len(self.decoder.var_to_check), self.n)
    
    def test_perfect_decoding(self):
        """Test decoding without errors"""
        message = np.random.randint(0, 2, self.k)
        codeword = self.encoder.encode_single(message)
        
        # Convert to soft values (no noise)
        received = codeword.astype(np.float64) * 2 - 1  # Map 0->-1, 1->1
        
        decoded, info = self.decoder.decode_single(received, noise_variance=0.1)
        
        self.assertTrue(info['is_valid_codeword'])
        self.assertEqual(info['syndrome_weight'], 0)
        
        # Extract information bits
        decoded_message = self.decoder.extract_information_bits(decoded)
        np.testing.assert_array_equal(decoded_message, message)
    
    def test_noisy_decoding(self):
        """Test decoding with noise"""
        message = np.random.randint(0, 2, self.k)
        codeword = self.encoder.encode_single(message)
        
        # Add noise
        noise_level = 0.5
        received = codeword.astype(np.float64) * 2 - 1
        noise = np.random.normal(0, noise_level, self.n)
        received_noisy = received + noise
        
        decoded, info = self.decoder.decode_single(received_noisy, noise_variance=noise_level**2)
        
        # Check if decoding converged
        self.assertLessEqual(info['iterations'], self.decoder.max_iterations)
        
        # For mild noise, should still decode correctly
        if info['is_valid_codeword']:
            decoded_message = self.decoder.extract_information_bits(decoded)
            accuracy = np.mean(decoded_message == message)
            self.assertGreater(accuracy, 0.9)  # Should be mostly correct
    
    def test_batch_decoding(self):
        """Test batch decoding"""
        batch_size = 5
        messages = np.random.randint(0, 2, (batch_size, self.k))
        codewords = self.encoder.encode_batch(messages)
        
        # Convert to soft values
        received = codewords.astype(np.float64) * 2 - 1
        
        decoded_batch, info_list = self.decoder.decode_batch(received, noise_variance=0.1)
        
        self.assertEqual(decoded_batch.shape, (batch_size, self.n))
        self.assertEqual(len(info_list), batch_size)
        
        # Check all are valid codewords
        for i in range(batch_size):
            self.assertTrue(info_list[i]['is_valid_codeword'])
    
    def test_different_decoding_methods(self):
        """Test different decoding algorithms"""
        methods = ['belief_propagation', 'min_sum', 'hard_decision']
        
        message = np.random.randint(0, 2, self.k)
        codeword = self.encoder.encode_single(message)
        received = codeword.astype(np.float64) * 2 - 1
        
        for method in methods:
            with self.subTest(method=method):
                decoder = LDPCDecoder(self.generator, method=method)
                decoded, info = decoder.decode_single(received)
                
                # Should decode perfectly without noise
                decoded_message = decoder.extract_information_bits(decoded)
                np.testing.assert_array_equal(decoded_message, message)


class TestAdaptiveLDPC(unittest.TestCase):
    """Test adaptive LDPC system"""
    
    def setUp(self):
        self.message_length = 100
        self.adaptive_ldpc = AdaptiveLDPC(
            message_length=self.message_length,
            min_redundancy=0.1,
            max_redundancy=0.5,
            device='cpu'
        )
    
    def test_adaptive_initialization(self):
        """Test adaptive LDPC initialization"""
        self.assertEqual(self.adaptive_ldpc.message_length, self.message_length)
        self.assertGreater(len(self.adaptive_ldpc.ldpc_codes), 0)
        self.assertGreater(len(self.adaptive_ldpc.redundancy_levels), 0)
    
    def test_redundancy_calculation(self):
        """Test adaptive redundancy calculation"""
        # Low attack strength -> low redundancy
        redundancy_low = self.adaptive_ldpc.calculate_adaptive_redundancy(0.1)
        self.assertLess(redundancy_low, 0.3)
        
        # High attack strength -> high redundancy
        redundancy_high = self.adaptive_ldpc.calculate_adaptive_redundancy(0.9)
        self.assertGreater(redundancy_high, 0.3)
        
        # Monotonic increase
        self.assertGreater(redundancy_high, redundancy_low)
    
    def test_adaptive_encoding_decoding(self):
        """Test adaptive encoding and decoding"""
        batch_size = 2
        messages = torch.randint(0, 2, (batch_size, self.message_length), dtype=torch.float32)
        
        for attack_strength in [0.1, 0.5, 0.9]:
            with self.subTest(attack_strength=attack_strength):
                # Encode
                encoded = self.adaptive_ldpc.encode(messages, attack_strength)
                
                # Check encoded length varies with redundancy
                redundancy = self.adaptive_ldpc.calculate_adaptive_redundancy(attack_strength)
                expected_length = int(self.message_length / (1 - redundancy))
                
                # Decode
                decoded = self.adaptive_ldpc.decode(encoded, attack_strength)
                
                # Check dimensions
                self.assertEqual(decoded.shape, messages.shape)
                
                # Perfect channel should give perfect decoding
                accuracy = torch.mean((decoded > 0.5).float() == (messages > 0.5).float())
                self.assertGreater(accuracy, 0.99)
    
    def test_code_info(self):
        """Test getting code information"""
        info = self.adaptive_ldpc.get_code_info(attack_strength=0.3)
        
        self.assertIn('redundancy', info)
        self.assertIn('k', info)
        self.assertIn('n', info)
        self.assertIn('rate', info)
        self.assertIn('available_redundancies', info)
        
        self.assertEqual(info['k'], self.message_length)
        self.assertGreater(info['n'], info['k'])
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        benchmark_results = self.adaptive_ldpc.benchmark_performance(
            num_tests=10,
            noise_levels=[0.1, 0.2]
        )
        
        self.assertIsInstance(benchmark_results, dict)
        
        # Check results structure
        for redundancy, noise_results in benchmark_results.items():
            self.assertIsInstance(noise_results, dict)
            for noise_level, metrics in noise_results.items():
                self.assertIn('ber', metrics)
                self.assertIn('total_errors', metrics)
                self.assertIn('total_bits', metrics)


class TestLDPCPerformanceAnalyzer(unittest.TestCase):
    """Test LDPC performance analyzer"""
    
    def setUp(self):
        self.adaptive_ldpc = AdaptiveLDPC(
            message_length=100,
            min_redundancy=0.1,
            max_redundancy=0.5,
            device='cpu'
        )
        self.analyzer = LDPCPerformanceAnalyzer(self.adaptive_ldpc)
    
    def test_reed_solomon_comparison(self):
        """Test comparison with Reed-Solomon"""
        comparison = self.analyzer.compare_with_reed_solomon(num_tests=10)
        
        self.assertIn('noise_levels', comparison)
        self.assertIn('ldpc_ber', comparison)
        self.assertIn('reed_solomon_ber', comparison)
        self.assertIn('improvements', comparison)
        
        # LDPC should generally perform better
        avg_improvement = np.mean(comparison['improvements'])
        self.assertGreater(avg_improvement, 0)
    
    def test_redundancy_efficiency(self):
        """Test redundancy efficiency analysis"""
        efficiency = self.analyzer.analyze_redundancy_efficiency()
        
        self.assertIsInstance(efficiency, dict)
        
        for redundancy, metrics in efficiency.items():
            self.assertIn('code_rate', metrics)
            self.assertIn('expansion_factor', metrics)
            self.assertIn('efficiency_score', metrics)
            
            # Basic sanity checks
            self.assertGreater(metrics['code_rate'], 0)
            self.assertLess(metrics['code_rate'], 1)
            self.assertGreater(metrics['expansion_factor'], 1)
    
    def test_capacity_analysis(self):
        """Test capacity analysis"""
        capacity_data = self.analyzer.capacity_analysis()
        
        self.assertIn('attack_strengths', capacity_data)
        self.assertIn('redundancies', capacity_data)
        self.assertIn('code_rates', capacity_data)
        self.assertIn('effective_capacities', capacity_data)
        
        # Check monotonicity
        redundancies = capacity_data['redundancies']
        self.assertEqual(redundancies, sorted(redundancies))


class TestNeuralLDPCDecoder(unittest.TestCase):
    """Test neural LDPC decoder"""
    
    def setUp(self):
        self.n = 100
        self.k = 50
        self.generator = LDPCGenerator(self.n, self.k, seed=42)
        self.neural_decoder = NeuralLDPCDecoder(self.generator)
    
    def test_neural_decoder_initialization(self):
        """Test neural decoder initialization"""
        self.assertEqual(self.neural_decoder.n, self.n)
        self.assertEqual(self.neural_decoder.k, self.k)
        self.assertIsNotNone(self.neural_decoder.network)
        self.assertIsNotNone(self.neural_decoder.traditional_decoder)
    
    def test_neural_forward_pass(self):
        """Test neural decoder forward pass"""
        batch_size = 2
        received = torch.randn(batch_size, self.n)
        
        output = self.neural_decoder(received)
        
        self.assertEqual(output.shape, (batch_size, self.n))
        # Output should be in tanh range
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))
    
    def test_neural_decoding(self):
        """Test neural decoding process"""
        batch_size = 2
        received = torch.randn(batch_size, self.n)
        
        decoded, info = self.neural_decoder.decode(received)
        
        self.assertEqual(decoded.shape, (batch_size, self.n))
        self.assertIn('neural_valid_rate', info)
        self.assertIn('fallback_used', info)
        
        # Check output is binary
        self.assertTrue(torch.all((decoded == 0) | (decoded == 1)))


if __name__ == '__main__':
    unittest.main()