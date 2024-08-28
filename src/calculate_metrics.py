import pandas as pd
import numpy as np
import math
from bert_score import BERTScorer
from nltk.metrics.distance import edit_distance
from transformers import AutoTokenizer
from perplexity import Perplexity
import sys
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from ensemble_debertav3 import DebertaPredictor
from hirschberg import *
from icecream import ic
from rouge_score import rouge_scorer


class MetricsCalculator:
    def __init__(self,
                 semantic_similarity=True,
                 token_edit_distance=True,
                 perplexity=True,
                 rouge=True,
                 classifier_prediction=True,
                 batch_size=4,
                 classifier_fold=0,
                 device_map=None,
                 device='cuda:0'
                 ):
        self.semantic_similarity = semantic_similarity
        self.token_edit_distance = token_edit_distance
        self.perplexity = perplexity
        self.rouge = rouge
        self.classifier_prediction = classifier_prediction
        self.batch_size = batch_size
        self.device_map = device_map
        self.device = device

        if semantic_similarity:
            self.semantic_similarity_model = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, lang="en", batch_size=self.batch_size, device=self.device)

        if token_edit_distance:
            nlp = English()
            self.word_tokenizer = Tokenizer(nlp.vocab)

        if perplexity:
            self.perplexity_model = Perplexity(model_id='gpt2', device='gpu')

        if rouge:
           self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        if classifier_prediction:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.set_classifier_based_on_fold(classifier_fold)

    def set_classifier_based_on_fold(self, fold):
        self.classifier = DebertaPredictor(
            '/bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue/fold0/'+str(fold)+'/checkpoint-1800', self.tokenizer, self.batch_size, device=self.device, device_map=self.device_map)
        self.classifier_fold = fold

    def calculate_metrics(self, x, y):
        output = {}
        print('Calculating metrics...')

        if self.semantic_similarity:
            print('Calculating semantic similarity...')
            _, _, semantic_similarities = self.semantic_similarity_model.score(x, y)
            output['semantic_similarities'] = semantic_similarities.numpy()
            output['mean_semantic_similarity'] = np.mean(semantic_similarities.numpy())

        if self.token_edit_distance:
            print('Calculating token edit distance...')
            token_edit_distances = []

            def normalized_edit_similarity(m, d):
                # d : edit distance between the two strings
                # m : length of the shorter string
                if m == d:
                    return 0.0
                elif d == 0:
                    return 1.0
                else:
                    return (1.0 / math.exp(d / (m - d)))

            for i in range(len(x)):
                y_tokens = [token.text for token in self.word_tokenizer(y[i])]
                x_tokens = [token.text for token in self.word_tokenizer(x[i])]
                Z, W, S = Hirschberg(x_tokens,  y_tokens)
                num_edits = len([s for s in S if s != '<KEEP>'])
                # token_edit_distances.append(num_edits)
                normalized_num_edits = normalized_edit_similarity(len(S), num_edits)
                token_edit_distances.append(normalized_num_edits)
            output['token_edit_distances'] = token_edit_distances
            output['mean_token_edit_distance'] = np.mean(token_edit_distances)

        if self.perplexity:
            print('Calculating perplexity...')
            gt_perplexities = self.perplexity_model._compute(
                data=y, batch_size=self.batch_size)['perplexities']
            prompt_perplexities = self.perplexity_model._compute(
                data=x, batch_size=self.batch_size)['perplexities']
            output['gt_perplexities'] = gt_perplexities
            output['prompt_perplexities'] = prompt_perplexities
            output['mean_gt_perplexity'] = np.mean(gt_perplexities)
            output['mean_prompt_perplexity'] = np.mean(prompt_perplexities)

        if self.rouge:
            print('Calculating rouge...')
            rouge_scores = []
            for i in range(len(x)):
                rouge_scores.append(self.rouge_scorer.score(x[i], y[i]))
            output['rouge1_f_scores'] = [s['rouge1'].fmeasure for s in rouge_scores]
            output['rouge2_f_scores'] = [s['rouge2'].fmeasure for s in rouge_scores]
            output['rougeL_f_scores'] = [s['rougeL'].fmeasure for s in rouge_scores]
            output['mean_rouge1_f_score'] = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
            output['mean_rouge2_f_score'] = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
            output['mean_rougeL_f_score'] = np.mean([s['rougeL'].fmeasure for s in rouge_scores])

        if self.classifier_prediction:
            print('Calculating classifier prediction...')
            classifier_predictions = self.classifier.predict(x)
            classifier_predictions = np.exp(
                classifier_predictions) / np.sum(np.exp(classifier_predictions),
                                                 axis=1, keepdims=True)
            output['classifier_predictions_app'] = classifier_predictions[:, 0]
            output['classifier_predictions_inapp'] = classifier_predictions[:, 1]
            output['classifier_predictions'] = np.argmax(
                classifier_predictions, axis=1)
            output['classifier_predictions'] = [1.0 if x == 1 else 0.0 for x in output['classifier_predictions']]
            output['mean_classifier_prediction_app'] = np.mean(
                classifier_predictions[:, 0])
            output['mean_classifier_prediction_inapp'] = np.mean(
                classifier_predictions[:, 1])
            output['mean_classifier_prediction'] = np.mean(
                output['classifier_predictions'])
            print('Done calculating metrics.')

        return output


def evaluate_neutralization():
    app_df = pd.read_csv('../data/inappropriate_arguments_sample_100_appropriateness.csv')
    llam_ndf = pd.read_csv('../data/results-by-corpus/appropriateness/neutralization/llama.csv', sep="\\t", header=None)
    llam_ndf.columns = ['id', 'llama_neutralization']
    llama_df_ppo_10a_00ss = pd.read_csv(
        '../data/results-by-corpus/appropriateness/neutralization/llama_ppo_rewrite_appropriateness_llama-7b-harmonic-mean-10a-00ss.csv', sep="\t", header=None)
    llama_df_ppo_10a_00ss.columns = ['id', 'llama_neutralization']

    app_df['source'] = app_df['argument'].tolist()
    app_df['rewrite_llama'] = llam_ndf['llama_neutralization']
    app_df['rewrite_llama_ppo'] = llama_df_ppo_10a_00ss['llama_neutralization']

    metric_calculator = MetricsCalculator()
    llama_metrics = metric_calculator.calculate_metrics(app_df['rewrite_llama'].tolist(), app_df['source'].tolist())
    llama_ppo_metrics = metric_calculator.calculate_metrics(app_df['rewrite_llama_ppo'].tolist(), app_df['source'].tolist())

    task_dict = {'task': ['neutralization', 'neutralization'],
                 'approach': ['llama', 'llama_ppo']}

    for key, value in llama_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)
    for key, value in llama_ppo_metrics.items(): 
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)

    return task_dict


def evaluate_snippets():
    args_df = pd.read_csv("../data/inappropriate_arguments_sample_100_argsme.csv")
    llama_df = pd.read_csv("../data/results-by-corpus/argsme/summarization/llama.csv", sep="\\t", header=None)
    llama_df.columns = ['id', 'llama_gist']
    bart_df = pd.read_csv("../data/results-by-corpus/argsme/summarization/bart.csv")

    args_df['source'] = args_df['argument'].tolist()
    args_df['argsme_snippet'] = args_df['snippet'].tolist()
    args_df['bart_snippet'] = bart_df['bart_gist']
    args_df['llama_snippet'] = llama_df['llama_gist']

    metric_calculator = MetricsCalculator()
    argsme_metrics = metric_calculator.calculate_metrics(args_df['argsme_snippet'].tolist(), args_df['source'].tolist())
    llama_metrics = metric_calculator.calculate_metrics(args_df['llama_snippet'].tolist(), args_df['source'].tolist())
    bart_metrics = metric_calculator.calculate_metrics(args_df['bart_snippet'].tolist(), args_df['source'].tolist())

    task_dict = {'task': ['snippet_generation', 'snippet_generation', 'snippet_generation'],
                 'approach': ['argsme', 'bart', 'llama']}

    for key, value in argsme_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)
    for key, value in bart_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)
    for key, value in llama_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)

    return task_dict


def evaluate_snippets_2():
    app_df = pd.read_csv('../data/inappropriate_arguments_sample_100_argsme.csv')
    bart_df = pd.read_csv('../data/results-by-corpus/argsme/summarization/bart.csv')
    llama_df_ppo_10a_00ss = pd.read_csv(
        '../data/results-by-corpus/argsme/both/bart_summarized_and_neutralized.csv', sep="\\t", header=None)
    llama_df_ppo_10a_00ss.columns = ['id', 'llama_neutralization']

    app_df['source'] = app_df['argument'].tolist()
    app_df['bart_snippet'] = bart_df['bart_gist']
    app_df['llama_snippet'] = llama_df_ppo_10a_00ss['llama_neutralization']

    metric_calculator = MetricsCalculator()
    bart_metrics = metric_calculator.calculate_metrics(app_df['bart_snippet'].tolist(), app_df['source'].tolist())
    llama_metrics = metric_calculator.calculate_metrics(app_df['llama_snippet'].tolist(), app_df['source'].tolist())
    snippet_comparison_metrics = metric_calculator.calculate_metrics(app_df['llama_snippet'].tolist(), app_df['bart_snippet'].tolist())

    task_dict = {'task': ['both', 'both', 'both'],
                 'approach': ['bart', 'llama', 'snippet_comparison']}

    for key, value in bart_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)
    for key, value in llama_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)
    for key, value in snippet_comparison_metrics.items():
        if 'mean' in key:
            if key not in task_dict:
                task_dict[key] = []
            task_dict[key].append(value)

    return task_dict


if __name__ == '__main__':
    neutralization_dict = evaluate_neutralization()
    snippet_dict = evaluate_snippets()
    both_dict = evaluate_snippets_2()
    neutralization_df = pd.DataFrame.from_dict(neutralization_dict)
    snippet_df = pd.DataFrame.from_dict(snippet_dict)
    both_df = pd.DataFrame.from_dict(both_dict)
    eval_df = pd.concat([neutralization_df, snippet_df, both_df])
    # round numbers in dataframe to 2 decimals
    eval_df = eval_df.round(2)
    ic(eval_df)
    eval_df.to_csv('../data/automatic_evaluation_results.csv', index=False)
