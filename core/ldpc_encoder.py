#!/usr/bin/env python3
"""
LDPC Encoder
High-performance LDPC encoding with multiple methods
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple
import logging
from .ldpc_generator import LDPCGenerator


class LDPCEncoder:
    """High-performance LDPC encoder with multiple encoding strategies"""
    
    def __init__(self, ldpc_generator: LDPCGenerator, method: str = "systematic"):
        """
        Initialize LDPC encoder
        
        Args:
            ldpc_generator: LDPC generator instance
            method: Encoding method ('systematic', 'generator', 'sparse')
        """
        self.ldpc_gen = ldpc_generator
        self.method = method
        self.H = ldpc_generator.H
        self.n = ldpc_generator.n
        self.k = ldpc_generator.k
        self.r = ldpc_generator.r
        
        if self.H is None:
            raise ValueError("LDPC generator must have valid H matrix")
        
        # Pre-compute encoding matrices based on method
        self._setup_encoding_method()
        
        logging.info(f"LDPC Encoder initialized: method={method}, rate={self.k/self.n:.3f}")
    
    def _setup_encoding_method(self):
        """Setup encoding based on selected method"""
        
        if self.method == "systematic":
            self._setup_systematic_encoding()
        elif self.method == "generator":
            self._setup_generator_encoding()
        elif self.method == "sparse":
            self._setup_sparse_encoding()
        else:
            logging.warning(f"Unknown encoding method: {self.method}. Using systematic.")
            self.method = "systematic"
            self._setup_systematic_encoding()
    
    def _setup_systematic_encoding(self):
        """Setup systematic encoding: [message | parity]"""
        
        # Convert H to systematic form [P | I]
        self.H_sys = self.ldpc_gen.to_systematic_form()
        
        if self.H_sys is not None:
            # Extract P matrix for parity calculation
            self.P = self.H_sys[:, :self.k]
            self.encoding_ready = True
            logging.info("Systematic encoding setup complete")
        else:
            # Fallback to generator matrix method
            logging.warning("Systematic form failed, falling back to generator method")
            self.method = "generator"
            self._setup_generator_encoding()
    
    def _setup_generator_encoding(self):
        """Setup generator matrix encoding"""
        
        self.G = self.ldpc_gen.generate_generator_matrix()
        
        if self.G is not None:
            self.encoding_ready = True
            logging.info("Generator matrix encoding setup complete")
        else:
            # Fallback to sparse method
            logging.warning("Generator matrix failed, falling back to sparse method")
            self.method = "sparse"
            self._setup_sparse_encoding()
    
    def _setup_sparse_encoding(self):
        """Setup sparse matrix encoding (direct solve)"""
        
        # For sparse encoding, we'll use the original H matrix
        # and solve the linear system for parity bits
        self.encoding_ready = True
        logging.info("Sparse encoding setup complete")
    
    def encode_single(self, message_bits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Encode a single message
        
        Args:
            message_bits: Binary message of length k
            
        Returns:
            Encoded codeword of length n
        """
        
        # Convert to numpy if needed
        if isinstance(message_bits, torch.Tensor):
            message_bits = message_bits.cpu().numpy()
        
        # Ensure correct shape and type
        message_bits = np.array(message_bits, dtype=np.int32).flatten()
        
        if len(message_bits) != self.k:
            if len(message_bits) < self.k:
                # Pad with zeros
                padding = np.zeros(self.k - len(message_bits), dtype=np.int32)
                message_bits = np.concatenate([message_bits, padding])
            else:
                # Truncate
                message_bits = message_bits[:self.k]
        
        # Encode based on method
        if self.method == "systematic":
            return self._encode_systematic(message_bits)
        elif self.method == "generator":
            return self._encode_generator(message_bits)
        elif self.method == "sparse":
            return self._encode_sparse(message_bits)
        else:
            raise ValueError(f"Unknown encoding method: {self.method}")
    
    def _encode_systematic(self, message_bits: np.ndarray) -> np.ndarray:
        """Systematic encoding: codeword = [message | parity]"""
        
        if not hasattr(self, 'P') or self.P is None:
            raise RuntimeError("Systematic encoding not properly initialized")
        
        # Calculate parity bits: parity = P * message (mod 2)
        parity_bits = np.dot(self.P, message_bits) % 2
        
        # Systematic codeword: [message | parity]
        codeword = np.concatenate([message_bits, parity_bits])
        
        return codeword.astype(np.int32)
    
    def _encode_generator(self, message_bits: np.ndarray) -> np.ndarray:
        """Generator matrix encoding: codeword = message * G (mod 2)"""
        
        if not hasattr(self, 'G') or self.G is None:
            raise RuntimeError("Generator matrix encoding not properly initialized")
        
        # Matrix multiplication in GF(2)
        codeword = np.dot(message_bits, self.G) % 2
        
        return codeword.astype(np.int32)
    
    def _encode_sparse(self, message_bits: np.ndarray) -> np.ndarray:
        """Sparse encoding using direct linear system solving"""
        
        try:
            # For sparse encoding, we need to solve: H * [m|p]^T = 0
            # Where m is message and p is parity
            # This becomes: H[:,:k]*m + H[:,k:]*p = 0
            # So: p = (H[:,k:])^(-1) * H[:,:k] * m
            
            H_msg = self.H[:, :self.k]  # Message part
            H_par = self.H[:, self.k:]  # Parity part
            
            # Calculate syndrome for message part
            syndrome = np.dot(H_msg, message_bits) % 2
            
            # Solve for parity bits (simplified - assumes H_par is invertible)
            if H_par.shape[0] == H_par.shape[1]:
                try:
                    H_par_inv = np.linalg.inv(H_par) % 2
                    parity_bits = np.dot(H_par_inv, syndrome) % 2
                except np.linalg.LinAlgError:
                    # Fallback: use least squares
                    parity_bits = np.linalg.lstsq(H_par, syndrome, rcond=None)[0] % 2
            else:
                # Non-square case: use pseudo-inverse
                parity_bits = np.linalg.pinv(H_par) @ syndrome % 2
            
            parity_bits = parity_bits.astype(np.int32)
            
            # Systematic codeword
            codeword = np.concatenate([message_bits, parity_bits])
            
            return codeword
            
        except Exception as e:
            logging.error(f"Sparse encoding failed: {e}")
            # Fallback to simple repetition
            return self._fallback_encoding(message_bits)
    
    def _fallback_encoding(self, message_bits: np.ndarray) -> np.ndarray:
        """Fallback encoding method (simple repetition)"""
        
        logging.warning("Using fallback repetition encoding")
        
        repeat_factor = self.n // self.k
        remainder = self.n % self.k
        
        # Repeat message bits
        codeword = np.tile(message_bits, repeat_factor)
        
        # Add remainder bits
        if remainder > 0:
            codeword = np.concatenate([codeword, message_bits[:remainder]])
        
        return codeword.astype(np.int32)
    
    def encode_batch(self, message_batch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Encode a batch of messages
        
        Args:
            message_batch: Batch of messages [batch_size, k]
            
        Returns:
            Batch of codewords [batch_size, n]
        """
        
        # Convert to numpy if needed
        if isinstance(message_batch, torch.Tensor):
            message_batch = message_batch.cpu().numpy()
        
        message_batch = np.array(message_batch, dtype=np.int32)
        
        if len(message_batch.shape) == 1:
            message_batch = message_batch.reshape(1, -1)
        
        batch_size = message_batch.shape[0]
        encoded_batch = np.zeros((batch_size, self.n), dtype=np.int32)
        
        # Encode each message in the batch
        for i in range(batch_size):
            encoded_batch[i] = self.encode_single(message_batch[i])
        
        return encoded_batch
    
    def verify_encoding(self, message_bits: np.ndarray, codeword: np.ndarray) -> bool:
        """
        Verify that encoding is correct
        
        Args:
            message_bits: Original message
            codeword: Encoded codeword
            
        Returns:
            True if encoding is valid
        """
        
        # Check syndrome: H * codeword should be zero
        syndrome = np.dot(self.H, codeword) % 2
        
        if np.any(syndrome != 0):
            logging.error(f"Invalid codeword: syndrome = {np.sum(syndrome)} (should be 0)")
            return False
        
        # For systematic codes, check if message part matches
        if self.method == "systematic":
            if not np.array_equal(message_bits, codeword[:self.k]):
                logging.error("Systematic encoding failed: message part mismatch")
                return False
        
        return True
    
    def get_encoding_info(self) -> dict:
        """Get information about the encoding setup"""
        
        info = {
            'method': self.method,
            'n': self.n,
            'k': self.k,
            'rate': self.k / self.n,
            'ready': self.encoding_ready
        }
        
        if hasattr(self, 'P'):
            info['has_systematic_matrix'] = True
            info['parity_matrix_shape'] = self.P.shape
        
        if hasattr(self, 'G'):
            info['has_generator_matrix'] = True
            info['generator_matrix_shape'] = self.G.shape
        
        return info


class ParallelLDPCEncoder(LDPCEncoder):
    """Parallel LDPC encoder for high-throughput applications"""
    
    def __init__(self, ldpc_generator: LDPCGenerator, method: str = "systematic", 
                 num_threads: int = 4):
        """
        Initialize parallel LDPC encoder
        
        Args:
            ldpc_generator: LDPC generator instance
            method: Encoding method
            num_threads: Number of parallel threads
        """
        super().__init__(ldpc_generator, method)
        self.num_threads = num_threads
        
        # Setup parallel processing
        try:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.parallel_ready = True
            logging.info(f"Parallel encoder ready with {num_threads} threads")
        except ImportError:
            logging.warning("Parallel processing not available")
            self.parallel_ready = False
    
    def encode_batch_parallel(self, message_batch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Encode batch in parallel
        
        Args:
            message_batch: Batch of messages [batch_size, k]
            
        Returns:
            Batch of codewords [batch_size, n]
        """
        
        if not self.parallel_ready:
            return self.encode_batch(message_batch)
        
        # Convert to numpy
        if isinstance(message_batch, torch.Tensor):
            message_batch = message_batch.cpu().numpy()
        
        message_batch = np.array(message_batch, dtype=np.int32)
        batch_size = message_batch.shape[0]
        
        # Split batch into chunks for parallel processing
        chunk_size = max(1, batch_size // self.num_threads)
        chunks = [message_batch[i:i+chunk_size] for i in range(0, batch_size, chunk_size)]
        
        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self.encode_batch, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            results.append(future.result())
        
        # Concatenate results
        return np.concatenate(results, axis=0)
    
    def __del__(self):
        """Cleanup parallel resources"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)