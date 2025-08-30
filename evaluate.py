import argparse
import json
from utils.evaluation_helper import evaluate_across_models_across_datasets
import os
from datetime import datetime
from utils.metrics import compute_pooled_eer

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINTS_DIR = os.environ.get('DF_ARENA_CHECKPOINTS_DIR', '/data/code/df_arena_stuff/checkpoints/checkpoints')
PROTOCOL_FILES_DIR = os.environ.get('DF_ARENA_PROTOCOL_FILES_DIR', '/data/code/df_arena_stuff/protocol_files')

def main():
    parser = argparse.ArgumentParser(description='Speech DF Arena Toolkit')
    parser.add_argument('--protocol_files', type=str, required=True, nargs='+',
                        help='Space seperated protocol names OR "all" to evaluate everything in ./protocol_files/',
                        choices=['all'] + [f.split('.')[0] for f in os.listdir(PROTOCOL_FILES_DIR) if f.endswith('.csv')])
    parser.add_argument('--batch_size', type=int, default=32, required=True)
    parser.add_argument('--models', type=str, default='Space seperated list of supported models form ./Models or "all" to evaluate all models', 
                        required=True, nargs='+',
                        choices=['all'] + [f.split('.')[0] for f in os.listdir('./Models') if f.endswith('.py') and f != 'model_factory.py'])
    parser.add_argument('--fix_length', action='store_true', 
                   help='Trim audio to 4 seconds for evaluation. Used by almost all the open source models.')    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    results = {}
    results_avg = evaluate_across_models_across_datasets(args, TIMESTAMP, CHECKPOINTS_DIR, PROTOCOL_FILES_DIR)
    results_pooled = compute_pooled_eer(results_avg, TIMESTAMP, PROTOCOL_FILES_DIR)

    results['Average'] = results_avg
    results['Pooled'] = results_pooled
    # Save all metrics to a single JSON file

    metrics_json_path = os.path.join("logs", f"summary_{TIMESTAMP}.json")

    print(f"[blue] Summary of results: {results} [/blue]")
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()