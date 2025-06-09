from sklearn.base import ClassifierMixin, RegressorMixin
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
import utils

import re


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
        self.models = []
        
        for k in cfg:
            if k.startswith("pipeline"):
                pipeline = instantiate(cfg[k], _recursive_=False)
                self.pipelines.append(pipeline)
                
                # Below tries to name the pipline as 'pipe' + pipeline_num + '-' + 'model_name'
                match = re.search(r'pipeline(\d+)', k)
                pipeline_num = match.group(1) if match else "X"
                model = None
                if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
                    model = pipeline.named_steps['model']
                elif hasattr(pipeline, 'steps'):
                    model = pipeline.steps[-1][1]
                
                self.models.append(model)

                model_name = type(model).__name__ if model is not None else 'UnknownModel'
                pipeline_name = f"pipe{pipeline_num}-{model_name}"
                
                self.pipeline_names.append(pipeline_name)

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
            return self.build_evaluation()
    
    def build_evaluation(self):
        evaluator = utils.Evaluator(
            class_threshold=self.class_threshold,
            decision_threshold=self.decision_threshold,
            plot=self.plot
        )
        
        for pipeline, model, pipeline_name in zip(self.pipelines, self.models, self.pipeline_names):
            for corp, data in self.splits.items():
                y_train_bin = data['wer_train'] < self.cfg.params.class_threshold
                #pipeline.fit(data['X_train'], y_train_bin)
                #y_proba = pipeline.predict_proba(data['X_test'])
                
                if isinstance(model, ClassifierMixin):
                    y_train_fit = y_train_bin
                    pipeline.fit(data['X_train'], y_train_fit)
                    y_proba = pipeline.predict_proba(data['X_test'])[:,1]
                
                elif isinstance(model, RegressorMixin):
                    y_train_fit = data['wer_train']
                    pipeline.fit(data['X_train'], y_train_fit)
                    y_pred = pipeline.predict(data['X_test'])
                    y_proba = 1-y_pred # here i am saying that if regression gives 0.3 output then it mean 0.7 chance of being class 1
                else:
                    raise ValueError(f"Model in {pipeline_name} seems to be neither regressor not classifier")

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

            # Plot for the pipeline if we are not comparing with other pipeline     
            if self.plot and (not self.compare_pipelines):
                if self.compare_corpora:
                    evaluator.plot_pr_curves(pipeline=pipeline_name, plot_all_corps=True)
                    evaluator.plot_box(pipeline=pipeline_name, plot_all_corps=True)
                else:
                    evaluator.plot_pr_curves(pipeline=pipeline_name, corp=self.corp_to_plot)
                    evaluator.plot_box(pipeline=pipeline_name, corp=self.corp_to_plot)

            if not self.compare_pipelines:
                return
        
        if self.plot and self.compare_pipelines:
            # Plot    
            if self.compare_corpora:
                evaluator.plot_pr_curves(plot_all_corps=True, plot_all_pipelines=True)
                #evaluator.plot_box(plot_all_corps=True, plot_all_pipelines=True)
            else:
                evaluator.plot_pr_curves(corp=self.corp_to_plot, plot_all_pipelines=True)
                evaluator.plot_box(corp=self.corp_to_plot, plot_all_pipelines=True)