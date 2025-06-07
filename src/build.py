from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

class Builder:
    def __init__(self, cfg: DictConfig):
        # config
        self.cfg = cfg
        #print(OmegaConf.to_yaml(cfg))
        
        # Load data using DataLoader and DataSplitter
        DataLoader = get_class(cfg.data_loader)
        DataSplitter = get_class(cfg.data_splitter)

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

        self.plot = cfg.params.plot
        self.print_all_eval = cfg.params.print_all_eval
        self.class_threshold = self.cfg.params.class_threshold
        self.decision_threshold = self.cfg.params.decision_threshold
        self.compare_corpora = self.cfg.params.compare_corpora
        self.compare_pipelines = self.cfg.params.compare_pipelines
        self.corp_to_plot = self.cfg.params.corp_to_plot

    def build_and_compare(self):
        if len(self.pipelines) < 1:
            raise ValueError("No pipeline given. Add a pipeline configuration in `defaults` in config.yaml")
        elif getattr(self.cfg.params, "compare_pipelines", False) and len(self.pipelines) < 2:
            raise ValueError("Invalid configuration: compare_piplines is true but not enough pipelines given! Check config.yaml")
        else:
            build_evaluation(self)
    
    def build_evaluation(self):
        evaluator = utils.Evaluator(
            class_threshold=self.class_threshold,
            decision_threshold=self.decision_threshold,
            plot=self.plot,
            compare_corpora=self.compare_corpora,
            compare_pipelines=self.compare_pipelines
        )
        
        for pipeline, pipeline_name in zip(self.pipelines, self.pipeline_names):
            for corp, data in self.splits.items():
                y_train_bin = data['wer_train'] < self.cfg.params.class_threshold
                pipeline.fit(data['X_train'], y_train_bin)
                y_proba = pipeline.predict_proba(data['X_test'])

                evaluator.evaluate(
                    y_test=data['wer_test'],
                    y_proba=y_proba,
                    corp=corp,
                    pipeline=pipeline_name
                )

            # Print
            if self.print_all_eval:
                evaluator.print_metrics(pipeline=pipeline_name, print_all=True, print_average=True)
            else:
                evaluator.print_metrics(pipeline=pipeline_name, corp=self.corp_to_plot, print_average=True)

            # Plot    
            if self.plot and self.compare_corpora:
                evaluator.plot_pr_curves(pipeline=pipeline_name, plot_all_corps=True)
                evaluator.plot_box(pipeline=pipeline_name, plot_all_corps=True)
            elif self.plot:
                evaluator.plot_pr_curves(pipeline=pipeline_name, corp=self.corp_to_plot)
                evaluator.plot_box(pipeline=pipeline_name, corp=self.corp_to_plot)

            if not self.compare_pipelines:
                return
        
        if not self.compare_pipelines:
            return
        else:
            if self.print_all_eval:
                evaluator.print_metrics(pipeline=pipeline_name, print_all=True, print_average=True)
            else:
                evaluator.print_metrics(pipeline=pipeline_name, corp=self.corp_to_plot, print_average=True)

            # Plot    
            if self.plot and self.compare_corpora:
                evaluator.plot_pr_curves(pipeline=pipeline_name, plot_all_corps=True)
                evaluator.plot_box(pipeline=pipeline_name, plot_all_corps=True)
            elif self.plot:
                evaluator.plot_pr_curves(pipeline=pipeline_name, corp=self.corp_to_plot)
                evaluator.plot_box(pipeline=pipeline_name, corp=self.corp_to_plot)