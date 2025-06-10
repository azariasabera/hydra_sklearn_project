import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

class Evaluator:
    def __init__(self, class_threshold, decision_threshold, plot=False):
        self.class_threshold = class_threshold
        self.decision_threshold = decision_threshold
        self.plot = plot
        self.results = {}  # Store results as {'pipeline' : {'corp': results}}
        self.results_plot = {} # stores result for plot

    def evaluate(self, y_test, y_proba, corp=None, pipeline=None):
        y_pred = (y_proba > self.decision_threshold).astype(int)
        y_test_bin = (y_test < self.class_threshold).astype(int)

        wer_low = y_test[y_pred == 1]
        wer_high = y_test[y_pred == 0]

        mean_low_wer = float(np.mean(wer_low)) if len(wer_low) > 0 else 0
        median_low_wer = float(np.median(wer_low)) if len(wer_low) > 0 else 0

        precision = precision_score(y_test_bin, y_pred)
        recall = recall_score(y_test_bin, y_pred)
        f1 = f1_score(y_test_bin, y_pred)

        precisions, recalls, _ = precision_recall_curve(y_test_bin, y_proba)
        pr_auc = auc(recalls, precisions)

        result = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_low_wer': mean_low_wer,
            'median_low_wer': median_low_wer,
            'auc': pr_auc
        }
        result_plot = {
            'auc': pr_auc,
            'wer_low': wer_low,
            'wer_high': wer_high,
            'precisions': precisions,
            'recalls': recalls
        }
        
        if pipeline:
            if corp:
                if pipeline not in self.results:
                    self.results[pipeline] = {}
                    self.results_plot[pipeline] = {}
                self.results[pipeline][corp] = result
                self.results_plot[pipeline][corp] = result_plot
            else:
                raise ValueError("Corpus name ('corp') must be provided when calling evaluate!")
        else:
            raise ValueError("Pipeline name ('pipeline') must be provided when calling evaluate!")

    def print_metrics(self, corp=None, pipeline=None, print_average=False, print_all=False):
        if pipeline:
            if corp:
                res = self.results[pipeline][corp]
                print(f"Results of corpus {corp} for {pipeline}:")
                for k, v in res.items():
                    print(f"{k}: {v:.4f}")
            
            elif print_all:
                for corp in self.results[pipeline]:
                    res = self.results[pipeline][corp]
                    print(f"Results of corpus {corp} for {pipeline}:")
                    for k, v in res.items():
                        print(f"{k}: {v:.4f}")

            if print_average:
                results = self.results[pipeline]
                avg = {k: np.mean([m[k] for m in results.values()]) for k in next(iter(results.values()))}
                print(f'\nAverage metrics over all corpora for {pipeline}:')
                for k, v in avg.items():
                    print(f"{k}: {v:.4f}")
                print()
        else:
            raise ValueError("Pipeline name ('pipeline') must be provided when calling print_metrics!")


    def plot_pr_curves(self, corp=None, pipeline= None, plot_all_corps=False, plot_all_pipelines=False):

        if plot_all_pipelines:
            if plot_all_corps: # plots all corpus of a pipeline in one figure, each pipeline in different figure
                plt.figure(figsize=(8, 6))
                for pipeline in self.results_plot:
                    for corp in self.results_plot[pipeline]:
                        res = self.results_plot[pipeline][corp]
                        plt.plot(res['recalls'], res['precisions'], marker='.', label=f"{pipeline}-{corp} (AUC={res['auc']:.3f})")
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.grid()
                plt.show()

            elif corp: # plots all pipeline in one figure, with the selected corp
                plt.figure(figsize=(8, 6))
                for pipeline in self.results_plot:
                    res = self.results_plot[pipeline][corp]
                    plt.plot(res['recalls'], res['precisions'], marker='.', label=f"{pipeline}-{corp} (AUC={res['auc']:.3f})")
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.grid()
                plt.show()

            else:
                raise ValueError("When plot_all_piplines, either a corp must be given or plot_all_corpus must be true!")
            
        elif pipeline:
            if plot_all_corps: # plots all corpus of the selected pipeline
                plt.figure(figsize=(8, 6))
                for corp in self.results_plot[pipeline]:
                    res = self.results_plot[pipeline][corp]
                    plt.plot(res['recalls'], res['precisions'], marker='.', label=f"{pipeline}-{corp} (AUC={res['auc']:.3f})")
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.grid()
                plt.show()

            elif corp: # plots the selected corpus for the selected pipeline
                plt.figure(figsize=(8, 6))
                res = self.results_plot[pipeline][corp]
                plt.plot(res['recalls'], res['precisions'], marker='.', label=f"{pipeline}-{corp} (AUC={res['auc']:.3f})")
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.grid()
                plt.show()

            else:
                raise ValueError("When pipeline given, either a corp must be given or plot_all_corpus must be true!")
        else:
            raise ValueError("either specify 'pipeline' or make 'plot_all_pipelines' true")

    def plot_box(self, corp=None, pipeline=None, plot_all_corps=False, plot_all_pipelines=False):
        """
        Flexible box plotter for various comparison scenarios.
        Each box shows WER for 'low' and 'high' per group.
        """
        if plot_all_pipelines:
            if plot_all_corps:
                return # would be too much
            elif corp:
                # Plot all pipelines for a specific corpus (all pipelines in one figure)
                data = []
                xticks = []
                for pipe in self.results_plot:
                    if corp in self.results_plot[pipe]:
                        res = self.results_plot[pipe][corp]
                        data.append(res['wer_low'])
                        data.append(res['wer_high'])
                        xticks.extend([f"{pipe}-Low", f"{pipe}-High"])
                plt.figure(figsize=(10, 6))
                box = plt.boxplot(data, patch_artist=True)
                for i, patch in enumerate(box['boxes']):
                    patch.set(facecolor='lightblue' if i % 2 == 0 else 'lightcoral')
                for patch in box['medians']:
                    patch.set(color='purple', linewidth=3)
                plt.axhline(y=self.class_threshold, color='green', linestyle='--', label='Class Threshold')
                plt.xticks(range(1, len(xticks) + 1), xticks, rotation=45)
                plt.title(f'WER Distribution - {corp} (all pipelines)')
                plt.ylabel('Word Error Rate (WER)')
                plt.legend()
                plt.grid(axis='y')
                plt.tight_layout()
                # Add jittered scatter
                for i, data_arr in enumerate(data):
                    x = np.random.normal(i + 1, 0.04, size=len(data_arr))
                    plt.plot(x, data_arr, 'o', alpha=0.5, markersize=6, color='gray')
                plt.show()
            else:
                raise ValueError("When plot_all_pipelines, either a corp must be given or plot_all_corps must be true!")

        elif pipeline:
            if plot_all_corps:
                # Plot all corpora for a specific pipeline (one figure)
                data = []
                xticks = []
                for corp_name in self.results_plot[pipeline]:
                    res = self.results_plot[pipeline][corp_name]
                    data.append(res['wer_low'])
                    data.append(res['wer_high'])
                    xticks.extend([f"{corp_name}-Low", f"{corp_name}-High"])
                plt.figure(figsize=(10, 6))
                box = plt.boxplot(data, patch_artist=True)
                for i, patch in enumerate(box['boxes']):
                    patch.set(facecolor='lightblue' if i % 2 == 0 else 'lightcoral')
                for patch in box['medians']:
                    patch.set(color='purple', linewidth=3)
                plt.axhline(y=self.class_threshold, color='green', linestyle='--', label='Class Threshold')
                plt.xticks(range(1, len(xticks) + 1), xticks, rotation=45)
                plt.title(f'WER Distribution - {pipeline}')
                plt.ylabel('Word Error Rate (WER)')
                plt.legend()
                plt.grid(axis='y')
                plt.tight_layout()
                # Add jittered scatter
                for i, data_arr in enumerate(data):
                    x = np.random.normal(i + 1, 0.04, size=len(data_arr))
                    plt.plot(x, data_arr, 'o', alpha=0.5, markersize=6, color='gray')
                plt.show()
            elif corp:
                # Plot one corpus for one pipeline (one figure)
                res = self.results_plot[pipeline][corp]
                data = [res['wer_low'], res['wer_high']]
                xticks = [f"{corp}-Low", f"{corp}-High"]
                plt.figure(figsize=(10, 6))
                box = plt.boxplot(data, patch_artist=True)
                for i, patch in enumerate(box['boxes']):
                    patch.set(facecolor='lightblue' if i == 0 else 'lightcoral')
                for patch in box['medians']:
                    patch.set(color='purple', linewidth=3)
                plt.axhline(y=self.class_threshold, color='green', linestyle='--', label='Class Threshold')
                plt.xticks([1, 2], xticks, rotation=45)
                plt.title(f'WER Distribution - {pipeline} - {corp}')
                plt.ylabel('Word Error Rate (WER)')
                plt.legend()
                plt.grid(axis='y')
                plt.tight_layout()
                # Add jittered scatter
                for i, data_arr in enumerate(data):
                    x = np.random.normal(i + 1, 0.04, size=len(data_arr))
                    plt.plot(x, data_arr, 'o', alpha=0.5, markersize=6, color='gray')
                plt.show()
            else:
                raise ValueError("When pipeline given, either a corp must be given or plot_all_corps must be true!")
        else:
            raise ValueError("Either specify 'pipeline' or make 'plot_all_pipelines' true")