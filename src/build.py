import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


class Builder:
    def __init__(self, cfg: DictConfig):
        # config
        self.cfg = cfg
        #print(OmegaConf.to_yaml(cfg))
        
        # Load data using DataLoader and DataSplitter
        DataLoader = hydra.utils.get_class(cfg.data_loader)
        DataSplitter = hydra.utils.get_class(cfg.data_splitter)

        self.data_loader = DataLoader(config=cfg, csv_file=cfg.paths.main_path)
        self.data_splitter = DataSplitter(config=cfg)

        df = self.data_loader.load_data()
        self.X, self.y, self.corpus = self.data_loader.extract_data(df)
        self.splits = self.data_splitter.loocv_corpus_split(self.X, self.y, self.corpus)

        # Build all pipelines listed in config
        self.pipelines = []
        self.pipeline_names = []
        
        for k in cfg:
            if k.startswith("pipeline"):
                self.pipelines.append(instantiate(cfg[k], _recursive_=False)) # False as each step is instantiated in pipeline.py
                self.pipeline_names.append(k)
                print(type(self.pipelines[-1]))

        self.plot = cfg.params.plot
        self.print_eval = cfg.params.print_eval

    def build_and_compare(self):
        if getattr(self.cfg.params, "compare_corpora", False): # better than 'if self.cfg.params.compare_corpora':
            self._compare_corpora()
        elif (len(self.pipelines) > 1 and getattr(self.cfg.params, "compare_pipelines", False)):
            self._compare_pipelines()
        elif len(self.pipelines) >= 1: # takes the first pipeline only
            self._single_pipeline_workflow()
        else:
            raise ValueError("Invalid configuration: at least one pipeline must be defined.")
        
    def _compare_pipelines(self):
        print('compared pipelines')

    def _single_pipeline_workflow(self):
        print('evaluated single pipeline')
        for corp, data in self.splits.items():
            y_bin = data['wer_train'] < 0.3
            self.pipelines[0].fit(data['X_train'], y_bin)
            y_pred = self.pipelines[0].predict_proba(data['X_test'])
            print(y_pred[:1])

    def _compare_corpora(self):
        print('compared corpora')