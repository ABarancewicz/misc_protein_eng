import logging
import shutil
from pathlib import Path
import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1
from chai_lab.chai1 import run_inference
import sys
import os
import torch



### Script to run Chai prediction on directory of pdbs
# Usage: chai_pdb_dir.py <pdb_dir> <output_dir> <processed_dir>
# pdb_dir: input pdbs
# output_dir: chai results
# processed_dir: move processed pdbs to here
# scores saved in results.txt

# Note: Currently using cuda device 0

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Note: Currently using msa server




def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_sequences_from_pdb(pdb_file):
    """Extract sequences from all chains in a PDB file"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residues = [res for res in chain if PDB.is_aa(res)]
            if residues:
                sequence = ''.join([seq1(res.get_resname()) for res in residues])
                sequences[chain_id] = sequence
    
    return sequences

def write_fasta(sequences, output_path, pdb_name):
    """Write sequences to a FASTA file in Chai format"""
    with open(output_path, 'w') as f:
        for chain_id, sequence in sequences.items():
            header = f">protein|name={pdb_name}_chain_{chain_id}"
            f.write(f"{header}\n{sequence}\n")

def run_chai_prediction(fasta_path, output_dir):
    """Run Chai prediction on the FASTA file"""
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=2,
        num_diffn_timesteps=150,
        seed=42,
        device="cuda:0",
        use_esm_embeddings=True,
        use_msa_server=True,
    )
    return candidates

def append_results(results_file, pdb_name, sequences, candidates):
    """Append results to the results.txt file"""
    with open(results_file, 'a') as f:
        f.write(f"\nResults for {pdb_name}:\n")
        f.write("-" * 50 + "\n")
        
        # Write sequences
        f.write("Sequences:\n")
        for chain_id, sequence in sequences.items():
            f.write(f"Chain {chain_id}: {sequence}\n")
        
        # Write scores
        f.write("\nScores:\n")
        agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
        for i, score in enumerate(agg_scores):
            f.write(f"Model {i+1} aggregate score: {score:.4f}\n")
        
        # Load and write detailed scores for best model
        scores = np.load(candidates.output_dir.joinpath("scores.model_idx_2.npz"))
        f.write("\nDetailed scores for best model:\n")
        for key, value in scores.items():
            if isinstance(value, np.ndarray):
                avg_value = np.mean(value)
                f.write(f"{key}: {avg_value:.4f}\n")
        f.write("\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <pdb_dir> <output_dir> <processed_dir>")
        sys.exit(1)

    pdb_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    processed_dir = Path(sys.argv[3])

    # Create output directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    setup_logging()
    results_file = output_dir / "results.txt"

    # Process each PDB file
    for pdb_file in pdb_dir.glob("*.pdb"):
        logging.info(f"Processing {pdb_file.name}")
        
        try:
            # Extract sequences
            sequences = extract_sequences_from_pdb(pdb_file)
            if not sequences:
                logging.warning(f"No protein sequences found in {pdb_file.name}")
                continue

            # Create fasta file
            fasta_path = output_dir / f"{pdb_file.stem}.fasta"
            write_fasta(sequences, fasta_path, pdb_file.stem)

            # Run Chai prediction
            prediction_output_dir = output_dir / f"{pdb_file.stem}_prediction"
            if prediction_output_dir.exists():
                shutil.rmtree(prediction_output_dir)
            prediction_output_dir.mkdir()
            
            candidates = run_chai_prediction(fasta_path, prediction_output_dir)

            # Append results
            append_results(results_file, pdb_file.stem, sequences, candidates)

            # Move processed PDB file
            shutil.move(str(pdb_file), str(processed_dir / pdb_file.name))
            
            logging.info(f"Successfully processed {pdb_file.name}")

        except Exception as e:
            logging.error(f"Error processing {pdb_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
