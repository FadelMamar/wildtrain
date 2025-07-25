import itertools
import os
from typing import Any, Dict, List
from omegaconf import OmegaConf, DictConfig, ListConfig
from .classification_trainer import ClassifierTrainer

class ClassifierSweeper:
    def __init__(self, sweep_config_path: str):
        self.sweep_config_path = sweep_config_path
        self.sweep_cfg = OmegaConf.load(sweep_config_path)
        self.base_cfg_path = os.path.join(os.path.dirname(sweep_config_path), self.sweep_cfg.base_config)
        self.base_cfg = OmegaConf.load(self.base_cfg_path)
        self.sweep_params = self.sweep_cfg.sweep.parameters
        self.strategy = self.sweep_cfg.sweep.get('strategy', 'grid')
        self.num_runs = self.sweep_cfg.sweep.get('num_runs', 10)
        self.seed = self.sweep_cfg.sweep.get('seed', 42)
        self.output_dir = self.sweep_cfg.output_dir if hasattr(self.sweep_cfg, 'output_dir') else 'runs/classification_sweeps'
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.sweep_params.keys())
        values = [self.sweep_params[k] for k in keys]
        combos = list(itertools.product(*values))
        param_dicts = []
        for combo in combos:
            d = {}
            for k, v in zip(keys, combo):
                d[k] = v
            param_dicts.append(d)
        return param_dicts

    def _set_nested(self, cfg, key, value):
        parts = key.split('.')
        sub = cfg
        for p in parts[:-1]:
            sub = sub[p]
        sub[parts[-1]] = value

    def run(self):
        param_combos = self._generate_param_combinations()
        results = []
        for i, params in enumerate(param_combos):
            cfg = OmegaConf.load(self.base_cfg_path)
            for k, v in params.items():
                self._set_nested(cfg, k, v)
            run_name = f"sweep_run_{i+1}"
            mlflow_cfg = OmegaConf.select(cfg, 'mlflow')
            if mlflow_cfg is None:
                cfg.mlflow = OmegaConf.create({})
            cfg.mlflow['run_name'] = run_name
            out_dir = os.path.join(self.output_dir, run_name)
            os.makedirs(out_dir, exist_ok=True)
            checkpoint_cfg = OmegaConf.select(cfg, 'checkpoint')
            if checkpoint_cfg is None:
                cfg.checkpoint = OmegaConf.create({})
            cfg.checkpoint['dirpath'] = out_dir
            print(f"[Sweep] Running {run_name} with params: {params}")
            # Ensure only DictConfig is passed
            if isinstance(cfg, ListConfig):
                cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            trainer = ClassifierTrainer(cfg)
            trainer.run()
            result = {
                'run_name': run_name,
                'params': params,
                'best_model_path': getattr(trainer, 'best_model_path', None)
            }
            results.append(result)
        OmegaConf.save({'results': results}, os.path.join(self.output_dir, 'sweep_results.yaml'))
