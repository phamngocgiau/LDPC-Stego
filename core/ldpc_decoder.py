#!/usr/bin/env python3
"""
LDPC Decoder
Advanced LDPC decoding with belief propagation and neural enhancement
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional, Tuple, Dict, Any
import logging
from .ldpc_generator import LDPCGenerator


class LDPCDecoder:
    """LDPC decoder with multiple decoding algorithms"""
    
    def __init__(self, ldpc_generator: LDPCGenerator, 
                 method: str = "belief_propagation",
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6):
        """
        Initialize LDPC decoder
        
        Args:
            ldpc_generator: LDPC generator instance
            method: Decoding method ('belief_propagation', 'min_sum', 'hard_decision')
            max_iterations: Maximum BP iterations
            convergence_threshold: Convergence threshold
        """
        self.ldpc_gen = ldpc_generator
        self.method = method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.H = ldpc_generator.H
        self.n = ldpc_generator.n
        self.k = ldpc_generator.k
        self.r = ldpc_generator.r
        
        if self.H is None:
            raise ValueError("LDPC generator must have valid H matrix")
        
        # Pre-compute decoder structures
        self._setup_decoder()
        
        logging.info(f"LDPC Decoder initialized: method={method}, max_iter={max_iterations}")
    
    def _setup_decoder(self):
        """Setup decoder data structures"""
        
        # Find non-zero positions in H matrix
        self.check_to_var = []  # For each check node, list of connected variable nodes
        self.var_to_check = []  # For each variable node, list of connected check nodes
        
        # Build adjacency lists
        for check in range(self.r):
            connected_vars = np.where(self.H[check, :] == 1)[0].tolist()
            self.check_to_var.append(connected_vars)
        
        for var in range(self.n):
            connected_checks = np.where(self.H[:, var] == 1)[0].tolist()
            self.var_to_check.append(connected_checks)
        
        # Pre-compute degrees
        self.var_degrees = [len(checks) for checks in self.var_to_check]
        self.check_degrees = [len(vars) for vars in self.check_to_var]
        
        logging.info(f"Decoder graph: avg var degree = {np.mean(self.var_degrees):.2f}, "
                    f"avg check degree = {np.mean(self.check_degrees):.2f}")
    
    def decode_single(self, received_symbols: Union[np.ndarray, torch.Tensor], 
                     noise_variance: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Decode a single received codeword
        
        Args:
            received_symbols: Received noisy symbols (soft values)
            noise_variance: Channel noise variance
            
        Returns:
            Tuple of (decoded_bits, decoding_info)
        """
        
        # Convert to numpy if needed
        if isinstance(received_symbols, torch.Tensor):
            received_symbols = received_symbols.cpu().numpy()
        
        received_symbols = np.array(received_symbols, dtype=np.float64)
        
        # Ensure correct length
        if len(received_symbols) != self.n:
            if len(received_symbols) < self.n:
                # Pad with zeros
                padding = np.zeros(self.n - len(received_symbols))
                received_symbols = np.concatenate([received_symbols, padding])
            else:
                # Truncate
                received_symbols = received_symbols[:self.n]
        
        # Decode based on method
        if self.method == "belief_propagation":
            return self._decode_belief_propagation(received_symbols, noise_variance)
        elif self.method == "min_sum":
            return self._decode_min_sum(received_symbols, noise_variance)
        elif self.method == "hard_decision":
            return self._decode_hard_decision(received_symbols)
        else:
            raise ValueError(f"Unknown decoding method: {self.method}")
    
    def _decode_belief_propagation(self, received_symbols: np.ndarray, 
                                  noise_variance: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standard belief propagation decoder"""
        
        # Initialize LLRs (Log-Likelihood Ratios)
        # LLR = log(P(bit=0)/P(bit=1))
        channel_llrs = 2 * received_symbols / noise_variance
        
        # Initialize messages
        var_to_check_msgs = {}  # Variable to check messages
        check_to_var_msgs = {}  # Check to variable messages
        
        # Initialize variable to check messages with channel LLRs
        for var in range(self.n):
            for check in self.var_to_check[var]:
                var_to_check_msgs[(var, check)] = channel_llrs[var]
        
        # Initialize check to variable messages
        for check in range(self.r):
            for var in self.check_to_var[check]:
                check_to_var_msgs[(check, var)] = 0.0
        
        # Belief propagation iterations
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            old_msgs = check_to_var_msgs.copy()
            
            # Update check to variable messages
            for check in range(self.r):
                connected_vars = self.check_to_var[check]
                
                for var in connected_vars:
                    # Collect all variable to check messages except from current variable
                    other_msgs = []
                    for other_var in connected_vars:
                        if other_var != var:
                            other_msgs.append(var_to_check_msgs[(other_var, check)])
                    
                    # Check node update: tanh rule
                    if len(other_msgs) > 0:
                        product = 1.0
                        for msg in other_msgs:
                            product *= np.tanh(msg / 2.0)
                        
                        # Avoid numerical issues
                        product = np.clip(product, -0.999, 0.999)
                        check_to_var_msgs[(check, var)] = 2 * np.arctanh(product)
                    else:
                        check_to_var_msgs[(check, var)] = 0.0
            
            # Update variable to check messages
            for var in range(self.n):
                connected_checks = self.var_to_check[var]
                
                for check in connected_checks:
                    # Sum channel LLR and all check to variable messages except current
                    total_llr = channel_llrs[var]
                    for other_check in connected_checks:
                        if other_check != check:
                            total_llr += check_to_var_msgs[(other_check, var)]
                    
                    var_to_check_msgs[(var, check)] = total_llr
            
            # Check convergence
            max_change = 0.0
            for key in check_to_var_msgs:
                change = abs(check_to_var_msgs[key] - old_msgs[key])
                max_change = max(max_change, change)
            
            convergence_history.append(max_change)
            
            if max_change < self.convergence_threshold:
                break
        
        # Final decision
        final_llrs = np.zeros(self.n)
        for var in range(self.n):
            # Sum channel LLR and all check to variable messages
            total_llr = channel_llrs[var]
            for check in self.var_to_check[var]:
                total_llr += check_to_var_msgs[(check, var)]
            final_llrs[var] = total_llr
        
        # Hard decisions
        decoded_bits = (final_llrs < 0).astype(np.int32)
        
        # Verification
        syndrome = np.dot(self.H, decoded_bits) % 2
        is_valid = np.all(syndrome == 0)
        
        decoding_info = {
            'iterations': iteration + 1,
            'converged': max_change < self.convergence_threshold,
            'is_valid_codeword': is_valid,
            'syndrome_weight': np.sum(syndrome),
            'final_llrs': final_llrs,
            'convergence_history': convergence_history
        }
        
        return decoded_bits, decoding_info
    
    def _decode_min_sum(self, received_symbols: np.ndarray, 
                       noise_variance: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Min-sum approximation of belief propagation"""
        
        # Initialize LLRs
        channel_llrs = 2 * received_symbols / noise_variance
        
        # Initialize messages
        var_to_check_msgs = {}
        check_to_var_msgs = {}
        
        # Initialize variable to check messages
        for var in range(self.n):
            for check in self.var_to_check[var]:
                var_to_check_msgs[(var, check)] = channel_llrs[var]
        
        # Initialize check to variable messages
        for check in range(self.r):
            for var in self.check_to_var[check]:
                check_to_var_msgs[(check, var)] = 0.0
        
        # Min-sum iterations
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            old_msgs = check_to_var_msgs.copy()
            
            # Update check to variable messages (min-sum rule)
            for check in range(self.r):
                connected_vars = self.check_to_var[check]
                
                for var in connected_vars:
                    other_msgs = []
                    for other_var in connected_vars:
                        if other_var != var:
                            other_msgs.append(var_to_check_msgs[(other_var, check)])
                    
                    if len(other_msgs) > 0:
                        # Min-sum update
                        signs = [1 if msg >= 0 else -1 for msg in other_msgs]
                        abs_msgs = [abs(msg) for msg in other_msgs]
                        
                        overall_sign = np.prod(signs)
                        min_magnitude = min(abs_msgs)
                        
                        check_to_var_msgs[(check, var)] = overall_sign * min_magnitude * 0.75  # Scaling factor
                    else:
                        check_to_var_msgs[(check, var)] = 0.0
            
            # Update variable to check messages
            for var in range(self.n):
                connected_checks = self.var_to_check[var]
                
                for check in connected_checks:
                    total_llr = channel_llrs[var]
                    for other_check in connected_checks:
                        if other_check != check:
                            total_llr += check_to_var_msgs[(other_check, var)]
                    
                    var_to_check_msgs[(var, check)] = total_llr
            
            # Check convergence
            max_change = max(abs(check_to_var_msgs[key] - old_msgs[key]) 
                           for key in check_to_var_msgs)
            convergence_history.append(max_change)
            
            if max_change < self.convergence_threshold:
                break
        
        # Final decision
        final_llrs = np.zeros(self.n)
        for var in range(self.n):
            total_llr = channel_llrs[var]
            for check in self.var_to_check[var]:
                total_llr += check_to_var_msgs[(check, var)]
            final_llrs[var] = total_llr
        
        decoded_bits = (final_llrs < 0).astype(np.int32)
        
        # Verification
        syndrome = np.dot(self.H, decoded_bits) % 2
        is_valid = np.all(syndrome == 0)
        
        decoding_info = {
            'iterations': iteration + 1,
            'converged': max_change < self.convergence_threshold,
            'is_valid_codeword': is_valid,
            'syndrome_weight': np.sum(syndrome),
            'final_llrs': final_llrs,
            'convergence_history': convergence_history
        }
        
        return decoded_bits, decoding_info
    
    def _decode_hard_decision(self, received_symbols: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hard decision decoding with bit flipping"""
        
        # Initial hard decisions
        hard_bits = (received_symbols > 0.5).astype(np.int32)
        
        # Bit flipping algorithm
        for iteration in range(self.max_iterations):
            # Calculate syndrome
            syndrome = np.dot(self.H, hard_bits) % 2
            
            if np.all(syndrome == 0):
                # Valid codeword found
                break
            
            # Find bit with highest syndrome weight
            syndrome_weights = np.zeros(self.n)
            for var in range(self.n):
                for check in self.var_to_check[var]:
                    syndrome_weights[var] += syndrome[check]
            
            # Flip bit with highest weight
            if np.max(syndrome_weights) > 0:
                flip_bit = np.argmax(syndrome_weights)
                hard_bits[flip_bit] = 1 - hard_bits[flip_bit]
            else:
                break
        
        decoding_info = {
            'iterations': iteration + 1,
            'converged': np.all(syndrome == 0),
            'is_valid_codeword': np.all(syndrome == 0),
            'syndrome_weight': np.sum(syndrome),
            'final_llrs': None
        }
        
        return hard_bits, decoding_info
    
    def decode_batch(self, received_batch: Union[np.ndarray, torch.Tensor],
                    noise_variance: float = 1.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Decode a batch of received codewords
        
        Args:
            received_batch: Batch of received symbols [batch_size, n]
            noise_variance: Channel noise variance
            
        Returns:
            Tuple of (decoded_batch, decoding_info_list)
        """
        
        # Convert to numpy if needed
        if isinstance(received_batch, torch.Tensor):
            received_batch = received_batch.cpu().numpy()
        
        received_batch = np.array(received_batch, dtype=np.float64)
        
        if len(received_batch.shape) == 1:
            received_batch = received_batch.reshape(1, -1)
        
        batch_size = received_batch.shape[0]
        decoded_batch = np.zeros((batch_size, self.n), dtype=np.int32)
        decoding_info_list = []
        
        for i in range(batch_size):
            decoded_bits, info = self.decode_single(received_batch[i], noise_variance)
            decoded_batch[i] = decoded_bits
            decoding_info_list.append(info)
        
        return decoded_batch, decoding_info_list
    
    def extract_information_bits(self, decoded_codeword: np.ndarray) -> np.ndarray:
        """
        Extract information bits from decoded codeword
        
        Args:
            decoded_codeword: Decoded codeword of length n
            
        Returns:
            Information bits of length k
        """
        
        # For systematic codes, information bits are the first k bits
        if hasattr(self.ldpc_gen, 'method') and self.ldpc_gen.method == 'systematic':
            return decoded_codeword[:self.k]
        
        # For non-systematic codes, would need generator matrix
        # For now, assume systematic
        return decoded_codeword[:self.k]


class NeuralLDPCDecoder(nn.Module):
    """Neural network enhanced LDPC decoder"""
    
    def __init__(self, ldpc_generator: LDPCGenerator, 
                 hidden_dims: List[int] = [512, 1024, 512],
                 dropout_rate: float = 0.1):
        """
        Initialize neural LDPC decoder
        
        Args:
            ldpc_generator: LDPC generator instance
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.ldpc_gen = ldpc_generator
        self.n = ldpc_generator.n
        self.k = ldpc_generator.k
        
        # Neural network layers
        layers = []
        input_dim = self.n
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer for codeword
        layers.append(nn.Linear(input_dim, self.n))
        layers.append(nn.Tanh())  # Soft output
        
        self.network = nn.Sequential(*layers)
        
        # Traditional decoder for comparison/fallback
        self.traditional_decoder = LDPCDecoder(ldpc_generator, "belief_propagation")
        
    def forward(self, received_symbols: torch.Tensor) -> torch.Tensor:
        """
        Neural network forward pass
        
        Args:
            received_symbols: Received noisy symbols [batch_size, n]
            
        Returns:
            Soft decoded symbols [batch_size, n]
        """
        return self.network(received_symbols)
    
    def decode(self, received_symbols: torch.Tensor, 
              use_traditional_fallback: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode using neural network with optional traditional fallback
        
        Args:
            received_symbols: Received symbols
            use_traditional_fallback: Whether to use traditional decoder as fallback
            
        Returns:
            Tuple of (decoded_bits, decoding_info)
        """
        
        # Neural decoding
        with torch.no_grad():
            soft_output = self.forward(received_symbols)
        
        # Hard decisions
        hard_decisions = (soft_output > 0.0).int()
        
        # Verify codewords
        batch_size = received_symbols.size(0)
        valid_codewords = torch.zeros(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            syndrome = torch.matmul(
                torch.tensor(self.ldpc_gen.H, dtype=torch.float32),
                hard_decisions[i].float()
            ) % 2
            valid_codewords[i] = torch.all(syndrome == 0)
        
        # Use traditional decoder for invalid codewords if fallback enabled
        if use_traditional_fallback:
            invalid_indices = torch.where(~valid_codewords)[0]
            
            for idx in invalid_indices:
                received_np = received_symbols[idx].cpu().numpy()
                decoded_bits, _ = self.traditional_decoder.decode_single(received_np)
                hard_decisions[idx] = torch.tensor(decoded_bits, dtype=torch.int32)
        
        decoding_info = {
            'neural_valid_rate': torch.mean(valid_codewords.float()).item(),
            'fallback_used': use_traditional_fallback and torch.any(~valid_codewords),
            'soft_output': soft_output
        }
        
        return hard_decisions, decoding_info
    
    def train_decoder(self, train_loader, num_epochs: int = 100, 
                     lr: float = 1e-3, device: str = 'cuda'):
        """
        Train the neural decoder
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Training device
        """
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (noisy_codewords, clean_codewords) in enumerate(train_loader):
                noisy_codewords = noisy_codewords.to(device)
                clean_codewords = clean_codewords.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(noisy_codewords)
                loss = criterion(outputs, clean_codewords)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        print("Neural decoder training completed")