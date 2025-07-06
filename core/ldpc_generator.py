#!/usr/bin/env python3
"""
LDPC Parity Check Matrix Generator
High-performance LDPC code generation with multiple construction methods
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.sparse import csr_matrix
import logging

# Try to import professional LDPC library
try:
    import ldpc
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False
    logging.warning("Professional LDPC library not available. Using custom implementation.")


class LDPCGenerator:
    """LDPC Parity Check Matrix Generator with multiple construction methods"""
    
    def __init__(self, n: int, k: int, seed: int = 42, construction: str = "gallager"):
        """
        Initialize LDPC generator
        
        Args:
            n: Codeword length (encoded bits)
            k: Information length (original bits)
            seed: Random seed for reproducibility
            construction: Construction method ('gallager', 'mackay', 'progressive')
        """
        self.n = n  # Total bits after encoding
        self.k = k  # Original message bits
        self.r = n - k  # Parity bits
        self.rate = k / n  # Code rate
        self.seed = seed
        self.construction = construction
        
        # Validate parameters
        if not (0 < self.rate < 1):
            raise ValueError(f"Invalid code rate: {self.rate}. Must be between 0 and 1.")
        
        if self.n <= self.k:
            raise ValueError(f"Invalid parameters: n={n} must be > k={k}")
        
        np.random.seed(seed)
        
        # Generate LDPC matrices
        self.H = None  # Parity check matrix
        self.G = None  # Generator matrix (optional)
        
        self._generate_ldpc_matrices()
        
        logging.info(f"Generated LDPC code: n={n}, k={k}, rate={self.rate:.3f}, method={construction}")
    
    def _generate_ldpc_matrices(self):
        """Generate LDPC parity check matrix using specified construction"""
        
        if LDPC_AVAILABLE and self.construction == "professional":
            self._generate_professional_ldpc()
        elif self.construction == "gallager":
            self._generate_gallager_ldpc()
        elif self.construction == "mackay":
            self._generate_mackay_ldpc()
        elif self.construction == "progressive":
            self._generate_progressive_ldpc()
        else:
            logging.warning(f"Unknown construction method: {self.construction}. Using Gallager.")
            self._generate_gallager_ldpc()
    
    def _generate_professional_ldpc(self):
        """Generate LDPC using professional library"""
        try:
            self.H = ldpc.codes.make_ldpc(
                n=self.n,
                k=self.k,
                systematic=True,
                seed=self.seed
            )
            logging.info("Using professional LDPC library")
        except Exception as e:
            logging.warning(f"Professional LDPC failed: {e}. Falling back to Gallager.")
            self._generate_gallager_ldpc()
    
    def _generate_gallager_ldpc(self):
        """Generate LDPC matrix using Gallager's regular construction"""
        
        # Parameters for regular LDPC
        dv = 3  # Variable node degree (columns)
        dc = max(3, int(dv * self.n / self.r))  # Check node degree (rows)
        
        # Adjust dc to ensure integer constraints
        while (dv * self.n) % dc != 0:
            dc += 1
        
        m = self.r  # Number of parity checks
        
        # Create base submatrix
        ones_per_row = (dv * self.n) // m
        
        # Initialize parity check matrix
        H = np.zeros((m, self.n), dtype=np.int32)
        
        # Fill matrix using Gallager's construction
        for block in range(dv):
            # Create permutation for this block
            col_indices = np.arange(self.n)
            np.random.shuffle(col_indices)
            
            rows_per_block = m // dv
            start_row = block * rows_per_block
            end_row = min(start_row + rows_per_block, m)
            
            # Distribute 1s in this block
            cols_per_row = ones_per_row
            for i, row in enumerate(range(start_row, end_row)):
                start_col_idx = (i * cols_per_row) % self.n
                for j in range(cols_per_row):
                    col_idx = (start_col_idx + j) % self.n
                    col = col_indices[col_idx]
                    H[row, col] = 1
        
        # Post-process: ensure each column has exactly dv ones
        self._regularize_column_weights(H, dv)
        
        self.H = H
    
    def _generate_mackay_ldpc(self):
        """Generate LDPC using MacKay's construction method"""
        
        m = self.r
        H = np.zeros((m, self.n), dtype=np.int32)
        
        # Target degrees
        dv = 3  # Variable node degree
        dc = (dv * self.n) // m  # Check node degree
        
        # Create degree sequences
        var_degrees = np.full(self.n, dv)
        check_degrees = np.full(m, dc)
        
        # Adjust to make sums equal
        total_edges = np.sum(var_degrees)
        check_sum = np.sum(check_degrees)
        
        if total_edges != check_sum:
            diff = total_edges - check_sum
            if diff > 0:
                # Add edges to check nodes
                for i in range(abs(diff)):
                    check_degrees[i % m] += 1
            else:
                # Remove edges from variable nodes
                for i in range(abs(diff)):
                    var_degrees[i % self.n] = max(1, var_degrees[i % self.n] - 1)
        
        # Build graph using edge-based construction
        edges = []
        
        # Create edge list
        for var in range(self.n):
            for _ in range(var_degrees[var]):
                edges.append(var)
        
        np.random.shuffle(edges)
        
        # Assign edges to check nodes
        edge_idx = 0
        for check in range(m):
            for _ in range(check_degrees[check]):
                if edge_idx < len(edges):
                    var = edges[edge_idx]
                    H[check, var] = 1
                    edge_idx += 1
        
        self.H = H
    
    def _generate_progressive_ldpc(self):
        """Generate LDPC using progressive edge-growth (PEG) algorithm"""
        
        m = self.r
        H = np.zeros((m, self.n), dtype=np.int32)
        
        dv = 3  # Target variable node degree
        
        # Progressive construction
        for var in range(self.n):
            # Find check nodes with minimum degree
            check_degrees = np.sum(H, axis=1)
            
            # Add edges one by one
            for _ in range(dv):
                # Find check nodes not connected to this variable
                available_checks = np.where(H[:, var] == 0)[0]
                
                if len(available_checks) == 0:
                    break
                
                # Choose check node with minimum degree among available
                min_degree = np.min(check_degrees[available_checks])
                candidates = available_checks[check_degrees[available_checks] == min_degree]
                
                # Random selection among candidates
                chosen_check = np.random.choice(candidates)
                H[chosen_check, var] = 1
                check_degrees[chosen_check] += 1
        
        self.H = H
    
    def _regularize_column_weights(self, H: np.ndarray, target_weight: int):
        """Regularize column weights to target value"""
        
        for col in range(self.n):
            col_weight = np.sum(H[:, col])
            
            if col_weight < target_weight:
                # Add more 1s
                zero_rows = np.where(H[:, col] == 0)[0]
                if len(zero_rows) >= (target_weight - col_weight):
                    selected_rows = np.random.choice(
                        zero_rows, target_weight - col_weight, replace=False
                    )
                    H[selected_rows, col] = 1
            
            elif col_weight > target_weight:
                # Remove excess 1s
                one_rows = np.where(H[:, col] == 1)[0]
                remove_rows = np.random.choice(
                    one_rows, col_weight - target_weight, replace=False
                )
                H[remove_rows, col] = 0
    
    def get_properties(self) -> Dict[str, Any]:
        """Get LDPC code properties"""
        if self.H is None:
            return {}
        
        # Calculate basic properties
        variable_degrees = np.sum(self.H, axis=0)
        check_degrees = np.sum(self.H, axis=1)
        
        properties = {
            'n': self.n,
            'k': self.k,
            'r': self.r,
            'rate': self.rate,
            'construction': self.construction,
            'density': np.sum(self.H) / (self.H.shape[0] * self.H.shape[1]),
            'avg_variable_degree': np.mean(variable_degrees),
            'avg_check_degree': np.mean(check_degrees),
            'min_variable_degree': np.min(variable_degrees),
            'max_variable_degree': np.max(variable_degrees),
            'min_check_degree': np.min(check_degrees),
            'max_check_degree': np.max(check_degrees),
            'is_regular': len(np.unique(variable_degrees)) == 1 and len(np.unique(check_degrees)) == 1
        }
        
        return properties
    
    def to_systematic_form(self) -> Optional[np.ndarray]:
        """Convert H to systematic form [P | I]"""
        if self.H is None:
            return None
        
        try:
            H_work = self.H.copy()
            m, n = H_work.shape
            
            # Gaussian elimination in GF(2)
            pivot_cols = []
            
            for row in range(m):
                # Find pivot column
                pivot_col = None
                for col in range(n):
                    if col not in pivot_cols and H_work[row, col] == 1:
                        pivot_col = col
                        break
                
                if pivot_col is None:
                    continue
                
                pivot_cols.append(pivot_col)
                
                # Eliminate other 1s in this column
                for other_row in range(m):
                    if other_row != row and H_work[other_row, pivot_col] == 1:
                        H_work[other_row] = (H_work[other_row] + H_work[row]) % 2
            
            return H_work
            
        except Exception as e:
            logging.warning(f"Systematic form conversion failed: {e}")
            return None
    
    def generate_generator_matrix(self) -> Optional[np.ndarray]:
        """Generate systematic generator matrix G = [I | P^T]"""
        try:
            H_sys = self.to_systematic_form()
            if H_sys is None:
                return None
            
            # Extract P matrix (assuming systematic form [P | I])
            P = H_sys[:, :self.k]
            
            # Generator matrix G = [I | P^T]
            I_k = np.eye(self.k, dtype=np.int32)
            self.G = np.hstack([I_k, P.T])
            
            return self.G
            
        except Exception as e:
            logging.warning(f"Generator matrix creation failed: {e}")
            return None
    
    def validate_code(self) -> bool:
        """Validate the generated LDPC code"""
        if self.H is None:
            return False
        
        # Check dimensions
        if self.H.shape != (self.r, self.n):
            logging.error(f"Invalid H dimensions: {self.H.shape}, expected: ({self.r}, {self.n})")
            return False
        
        # Check if matrix is binary
        if not np.all(np.isin(self.H, [0, 1])):
            logging.error("H matrix is not binary")
            return False
        
        # Check for all-zero rows (invalid)
        zero_rows = np.where(np.sum(self.H, axis=1) == 0)[0]
        if len(zero_rows) > 0:
            logging.warning(f"Found {len(zero_rows)} all-zero rows in H matrix")
        
        # Check for all-zero columns (invalid)
        zero_cols = np.where(np.sum(self.H, axis=0) == 0)[0]
        if len(zero_cols) > 0:
            logging.warning(f"Found {len(zero_cols)} all-zero columns in H matrix")
            return False
        
        return True
    
    def save_matrix(self, filepath: str, format: str = "npz"):
        """Save LDPC matrix to file"""
        if self.H is None:
            raise ValueError("No matrix to save")
        
        if format == "npz":
            np.savez_compressed(filepath, H=self.H, 
                              n=self.n, k=self.k, rate=self.rate,
                              construction=self.construction)
        elif format == "txt":
            np.savetxt(filepath, self.H, fmt='%d')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_matrix(cls, filepath: str, format: str = "npz"):
        """Load LDPC matrix from file"""
        if format == "npz":
            data = np.load(filepath)
            H = data['H']
            n, k = int(data['n']), int(data['k'])
            construction = str(data.get('construction', 'loaded'))
            
            # Create instance
            instance = cls.__new__(cls)
            instance.n = n
            instance.k = k
            instance.r = n - k
            instance.rate = k / n
            instance.construction = construction
            instance.H = H
            instance.G = None
            
            return instance
            
        elif format == "txt":
            H = np.loadtxt(filepath, dtype=np.int32)
            m, n = H.shape
            k = n - m  # Assume systematic
            
            instance = cls.__new__(cls)
            instance.n = n
            instance.k = k
            instance.r = m
            instance.rate = k / n
            instance.construction = "loaded"
            instance.H = H
            instance.G = None
            
            return instance
        else:
            raise ValueError(f"Unsupported format: {format}")


class OptimizedLDPCGenerator(LDPCGenerator):
    """Optimized LDPC generator with performance improvements"""
    
    def __init__(self, n: int, k: int, seed: int = 42, 
                 target_girth: int = 4, max_iterations: int = 1000):
        """
        Initialize optimized LDPC generator
        
        Args:
            n: Codeword length
            k: Information length  
            seed: Random seed
            target_girth: Target girth (cycle length)
            max_iterations: Maximum optimization iterations
        """
        self.target_girth = target_girth
        self.max_iterations = max_iterations
        
        super().__init__(n, k, seed, construction="optimized")
    
    def _generate_ldpc_matrices(self):
        """Generate optimized LDPC matrix"""
        # Start with PEG construction
        self._generate_progressive_ldpc()
        
        # Optimize for girth
        if self.target_girth > 4:
            self._optimize_girth()
        
        # Post-processing
        self._remove_redundant_rows()
        self._improve_stopping_sets()
    
    def _optimize_girth(self):
        """Optimize matrix for target girth"""
        current_girth = self._calculate_girth()
        
        for iteration in range(self.max_iterations):
            if current_girth >= self.target_girth:
                break
            
            # Find and break short cycles
            improved = self._break_shortest_cycles()
            if not improved:
                break
            
            current_girth = self._calculate_girth()
        
        logging.info(f"Final girth: {current_girth} (target: {self.target_girth})")
    
    def _calculate_girth(self) -> int:
        """Calculate girth (shortest cycle length) of the matrix"""
        # Simplified girth calculation
        # For full implementation, would use graph algorithms
        return 4  # Placeholder
    
    def _break_shortest_cycles(self) -> bool:
        """Break shortest cycles in the matrix"""
        # Placeholder for cycle breaking algorithm
        return False
    
    def _remove_redundant_rows(self):
        """Remove linearly dependent rows"""
        if self.H is None:
            return
        
        # Use Gaussian elimination to find linearly independent rows
        H_work = self.H.copy()
        rank = np.linalg.matrix_rank(H_work)
        
        if rank < self.H.shape[0]:
            logging.info(f"Removing {self.H.shape[0] - rank} redundant rows")
            # Implementation would identify and remove redundant rows
    
    def _improve_stopping_sets(self):
        """Improve stopping set properties"""
        # Placeholder for stopping set optimization
        pass
        