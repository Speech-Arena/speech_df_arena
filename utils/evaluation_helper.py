import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.datamodule import DataModule
from Models.model_factory import ModelFactory
from utils.metrics import compute_metrics
from rich import print
import json
from contextlib import redirect_stdout
import os
from datetime import datetime

SEED = 42

pl.seed_everything(SEED, workers=True)


def evaluate_across_models_across_datasets(args, timestamp, checkpoints_dir, protocol_files_dir):

    batch_size = args.batch_size
    fix_length = args.fix_length
    num_workers = args.num_workers
    timestamp = timestamp
    checkpoints_dir = checkpoints_dir
    protocol_files_dir = protocol_files_dir

    results_dict = {}

    trainer = pl.Trainer(accelerator=args.device, enable_model_summary=True, logger=False)

    if args.models[0] == 'all':
        models = [i.split('.')[0] for i in os.listdir('./Models') if i != 'model_factory.py']
    else:
        models = args.models
        

    if args.protocol_files[0] == 'all':
        protocols = [i.split('.')[0] for i in os.listdir(protocol_files_dir)]
    else:
        protocols = args.protocol_files

    for model in models:
        results_dict[model] = {}

        model_path = [os.path.join(checkpoints_dir, i) for i in os.listdir(checkpoints_dir) if i.startswith(model)][0]
        _model = ModelFactory.get_model(model_name=model , model_path=model_path, 
                                        out_score_file_name='bla')

        for protocol in protocols:
            protocol_path = f'{protocol_files_dir}/{protocol}.csv'
            out_score_file = f'./scores/scores_{timestamp}/{model}/{protocol}.txt'
            _model.out_score_file_name = out_score_file

            if not os.path.exists(protocol_path):
                print(f"[red]Protocol file not found: {protocol}[/red]")
                continue

            log_file = f'./logs/logs_{timestamp}/{model}/{protocol}.log'

            os.makedirs(os.path.dirname(out_score_file), exist_ok=True)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            print(f"[bold green]Evaluating for {model} on {protocol}[/bold green]. Fix length to 4s -> {fix_length}")
            print(f"[blue]Score file: {out_score_file}[/blue]")
            print(f"[blue]Log file: {log_file}[/blue]")


            data_module = DataModule(batch_size, protocol_path, fix_length, args.num_workers)
            data_module.prepare_data()
            data_module.setup('test')


            with open(log_file, 'w') as f_log:
                with redirect_stdout(f_log):
                    trainer.test(_model, datamodule=data_module)

            metrics = compute_metrics(out_score_file, protocol_path)
            print(f"[cyan]Results for {model} on {protocol}: {metrics}[/cyan]")

            results_dict[model][protocol] = metrics

    return results_dict
