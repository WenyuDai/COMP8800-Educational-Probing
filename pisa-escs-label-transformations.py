# !pip install git+https://github.com/davidbau/baukit
# !pip install accelerate
# !pip install einops

import json
import pickle
import random
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as stats
import seaborn as sns
import torch
import transformers
from einops import rearrange
# from IPython.display import display, HTML
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baukit import Trace, TraceDict
from custom_llama import llama # modified code to access attention head outputs

warnings.filterwarnings("ignore")
device = 'cuda'

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

label = 'ESCS'

country_list = ['AUS', 'BRA', 'JPN', 'CHE']
country_name_map = {
    "AUS": "Australia",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "DEU": "Germany",
    "ESP": "Spain",
    "IDN": "Indonesia",
    "JPN": "Japan",
    "KOR": "South Korea",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "TUR": "Turkey",
    "UK":  "United Kingdom",
    "USA": "United States"
}

for country_code in country_list:
    loop = 5
    print(f'Start robust phase for country {country_code}')

    while loop <= 6:
        # """
        # Robustness
        # Step 1: label transformations
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affects subprocesses

        def compare_probes(transform_type):
            for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
                print(model_name)
                features, labels = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_features.pkl", 'rb'))
                if transform_type == 'random':
                    labels = np.random.permutation(labels)
                elif transform_type == 'sin':
                    labels = np.sin(labels * 10)
                elif transform_type == 'cube':
                    labels = np.power(labels, 3)
                kf = KFold(n_splits=2, shuffle=True, random_state=42)
                # max_performance = 0
                # max_ij = []
                max_performance = -np.inf
                max_ij = None
                for i in range(32):
                    for j in range(32):
                        performance = 0
                        for train_indices, test_indices in kf.split(range(features.shape[0])):
                            X_train = features[train_indices, 0, i, j, :]
                            X_test = features[test_indices, 0, i, j, :]
                            y_train = labels[train_indices]
                            y_test = labels[test_indices]

                            probe_model = Ridge(alpha=1, fit_intercept=True)
                            probe_model.fit(X_train, y_train)
                            y_pred = probe_model.predict(X_test)
                            performance += spearmanr(y_test, y_pred).statistic
                        # if max_performance < performance / 2:
                        #     max_performance = performance / 2
                        #     max_ij = [i, j]
                        score = performance / 2
                        if np.isfinite(score) and score > max_performance:
                            max_performance = score
                            max_ij = [i, j]

                print(transform_type, max_performance, max_ij)

                predicted = np.zeros(labels.shape)
                i, j = max_ij
                actual = labels

                for train_indices, test_indices in kf.split(range(features.shape[0])):
                    X_train = features[train_indices, 0, i, j, :]
                    X_test = features[test_indices, 0, i, j, :]
                    y_train = labels[train_indices]
                    y_test = labels[test_indices]

                    ## Keep the same as above
                    probe_model = Ridge(alpha=1, fit_intercept=True)
                    probe_model.fit(X_train, y_train)
                    predicted[test_indices] = probe_model.predict(X_test)

                # Plot with 90% opacity, larger fonts
                plt.figure(figsize=(6, 6))  # Keep figure square
                plt.scatter(predicted, actual, alpha=0.9)  # 90% opacity

                # Labels and title with larger font
                plt.xlabel("Predicted", fontsize=16)
                plt.ylabel("Actual", fontsize=16)
                plt.title("", fontsize=18)

                # Remove top and right spines
                ax = plt.gca()
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Increase tick font size
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                if transform_type != 'cube':
                    plt.xlim(predicted.min(), predicted.max())
                    plt.ylim(actual.min(), actual.max())
                
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.tight_layout()
                plt.savefig(
                    f"./report/figures/{country_code}_{model_name.replace('/','_')}_linearity_{transform_type}_{loop}.png",
                    dpi=300,
                    bbox_inches='tight'
                )

        for transform_type in ['original', 'random', 'sin', 'cube']:
            compare_probes(transform_type)

        loop += 1