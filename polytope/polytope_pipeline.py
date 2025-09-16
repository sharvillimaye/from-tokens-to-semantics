"""finalized polytope pipeline"""

from typing import Dict, Any
import pickle

from polytope.polytope_analysis import PolytopeAnalyzer


def load_activation_data(activation_path: str) -> Dict[str, Any]:
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)

    records = data['records']
    metadata = data['metadata']
    return {'records': records, 'metadata': metadata}


def run_polytope_pipeline(activation_path: str, output_path: str) -> Dict[str, Any]:
    analyzer = PolytopeAnalyzer()
    results = analyzer.run_checkpoint_file(
        checkpoint_path=activation_path,
        output_dir=output_path,
        parallel_layers=True,
    )
    return results
    