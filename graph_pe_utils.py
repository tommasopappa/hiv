import numpy as np
from scipy.sparse.linalg import eigsh
from rdkit import Chem
from typing import Optional


def smiles_to_adjacency_matrix(smiles: str) -> Optional[np.ndarray]:
    """Convert SMILES to adjacency matrix."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((n_atoms, n_atoms))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Undirected graph
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    return adj_matrix


def compute_graph_positional_encoding(adj_matrix: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Compute graph positional encodings from eigenvectors of the
    symmetrically normalized graph Laplacian.
    """
    n = adj_matrix.shape[0]

    # Compute degree matrix
    degree = np.sum(adj_matrix, axis=1)
    degree[degree == 0] = 1

    # D^(-1/2)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    # Symmetrically normalized Laplacian
    identity = np.eye(n)
    normalized_adj = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    laplacian = identity - normalized_adj

    # Compute eigenvectors
    if n < k:
        # If graph has fewer nodes than k, pad with zeros
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Pad eigenvectors to have k columns
        padded_eigenvectors = np.zeros((n, k))
        padded_eigenvectors[:, :n] = eigenvectors
        return padded_eigenvectors
    else:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
        return eigenvectors


def compute_graph_pe(smiles: str, k: int = 30) -> Optional[np.ndarray]:
    """
    Compute graph positional encoding for a SMILES string.
    
    Args:
        smiles: SMILES string
        k: Number of eigenvectors to use (dimension of PE)
    
    Returns:
        Array of shape (n_atoms, k) containing positional encodings
    """
    adj_matrix = smiles_to_adjacency_matrix(smiles)
    if adj_matrix is None:
        return None
    
    pe = compute_graph_positional_encoding(adj_matrix, k=k)
    return pe


def batch_compute_graph_pe(smiles_list: list, k: int = 30) -> tuple:
    """
    Process all SMILES and compute their graph PEs.
    
    Args:
        smiles_list: List of SMILES strings
        k: Number of eigenvectors to use
    
    Returns:
        Tuple of (list of PEs, maximum number of nodes)
    """
    graph_pes_list = []
    max_nodes = 0

    for smiles in smiles_list:
        adj_matrix = smiles_to_adjacency_matrix(smiles)
        if adj_matrix is not None:
            pe = compute_graph_positional_encoding(adj_matrix, k=k)
            graph_pes_list.append(pe)
            max_nodes = max(max_nodes, pe.shape[0])
        else:
            graph_pes_list.append(None)
    
    print(f"Maximum number of atoms in dataset: {max_nodes}")
    
    return graph_pes_list, max_nodes


def batch_compute_graph_pe_dict(smiles_list: list, k: int = 30) -> dict:
    """
    Compute graph PEs for a batch of SMILES and return as dictionary.
    
    Returns:
        Dictionary mapping SMILES to their PEs
    """
    pe_dict = {}
    max_nodes = 0
    
    for smiles in smiles_list:
        adj_matrix = smiles_to_adjacency_matrix(smiles)
        if adj_matrix is not None:
            pe = compute_graph_positional_encoding(adj_matrix, k=k)
            pe_dict[smiles] = pe
            max_nodes = max(max_nodes, pe.shape[0])
        else:
            pe_dict[smiles] = None
    
    print(f"Maximum number of atoms in dataset: {max_nodes}")
    
    return pe_dict


# Include the align_graph_pe_to_tokens function
import torch
import re


def generate_random_orthonormal_pe(n_vectors, dim=30):
    """Generate n_vectors random orthonormal vectors of dimension dim."""
    if n_vectors == 0:
        return np.zeros((0, dim))

    # Generate random matrix
    random_matrix = np.random.randn(dim, n_vectors)

    # Use QR decomposition to get orthonormal vectors
    Q, _ = np.linalg.qr(random_matrix)

    # Return first n_vectors columns (transposed to have shape (n_vectors, dim))
    return Q[:, :n_vectors].T


def parse_token_components(token):
    """
    Parse a token to identify atoms and characters.
    Returns: (atom_indices, n_characters)
    """
    # Common atom patterns in SMILES
    atom_pattern = r'(Cl|Br|Si|Mg|Ca|Fe|Al|Na|Li|[BCNOFPSKHIV])'

    # Find all atoms in the token
    atoms = re.findall(atom_pattern, token)

    # Count non-atom characters
    # Remove atoms from token to count remaining characters
    remaining = token
    for atom in atoms:
        remaining = remaining.replace(atom, '', 1)

    # Count actual characters (digits, +, -, =, #, etc.)
    n_characters = len([c for c in remaining])

    return atoms, n_characters


def align_graph_pe_to_tokens(smiles, graph_pe, tokenizer, max_length, random_seed=None):
    """
    Align graph PE to tokens, handling pure characters, atoms, and mixed tokens.

    Args:
        smiles: SMILES string
        graph_pe: Graph positional encoding for atoms (n_atoms, embedding_dim)
        tokenizer: Tokenizer object
        max_length: Maximum sequence length
        random_seed: Random seed for reproducible random PEs
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # Determine embedding dimension
    if graph_pe is not None:
        embedding_dim = graph_pe.shape[1]
    else:
        embedding_dim = 30  # Default fallback

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or graph_pe is None:
        return torch.zeros(max_length, embedding_dim)

    # Get tokens
    encoding = tokenizer(smiles, padding='max_length', max_length=max_length)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])

    # Initialize token PE
    token_pe = torch.zeros(max_length, embedding_dim)

    # Build atom mapping from SMILES
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_count = {symbol: 0 for symbol in set(atom_symbols)}
    atom_to_idx = {}

    for idx, symbol in enumerate(atom_symbols):
        atom_to_idx[(symbol, atom_count[symbol])] = idx
        atom_count[symbol] += 1

    # Reset atom count for tracking
    current_atom_count = {symbol: 0 for symbol in atom_count}

    # Process each token
    for i, token in enumerate(tokens):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        # Parse token components
        atoms_in_token, n_characters = parse_token_components(token)

        if len(atoms_in_token) == 0 and n_characters > 0:
            # Pure character token - use random orthonormal PE
            random_pes = generate_random_orthonormal_pe(n_characters, embedding_dim)
            token_pe[i] = torch.tensor(random_pes.sum(axis=0))

        elif len(atoms_in_token) > 0 and n_characters == 0:
            # Pure atom token(s) - use graph PE
            atom_pes = []
            for atom_symbol in atoms_in_token:
                if atom_symbol in current_atom_count:
                    atom_key = (atom_symbol, current_atom_count[atom_symbol])
                    if atom_key in atom_to_idx:
                        atom_idx = atom_to_idx[atom_key]
                        atom_pes.append(graph_pe[atom_idx])
                        current_atom_count[atom_symbol] += 1

            if atom_pes:
                # Average the PEs of all atoms in this token
                token_pe[i] = torch.tensor(np.sum(atom_pes, axis=0))

        elif len(atoms_in_token) > 0 and n_characters > 0:
            # Mixed token - combine atom PE and character PE
            # Get atom PEs
            atom_pes = []
            for atom_symbol in atoms_in_token:
                if atom_symbol in current_atom_count:
                    atom_key = (atom_symbol, current_atom_count[atom_symbol])
                    if atom_key in atom_to_idx:
                        atom_idx = atom_to_idx[atom_key]
                        atom_pes.append(graph_pe[atom_idx])
                        current_atom_count[atom_symbol] += 1

            # Get character PEs
            random_pes = generate_random_orthonormal_pe(n_characters, embedding_dim)

            # Combine: sum of atoms + sum of characters
            combined_pe = np.zeros(embedding_dim)
            if atom_pes:
                combined_pe += np.sum(atom_pes, axis=0)
            combined_pe += random_pes.sum(axis=0)

            token_pe[i] = torch.tensor(combined_pe)

    return token_pe
