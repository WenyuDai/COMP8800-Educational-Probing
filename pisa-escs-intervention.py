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



"""
Read before running the code:

Each step in this code file is independent and should not be run all at once. 
Only one step should be uncommented and run at a time.
"""



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
        # Steering
        # Step 1: Generate parquet and jsonl files to mark resource assumption scores
        """
        df = pd.read_parquet(f"./report/results/{country_code}_{label}_{loop}_intervention_raw.parquet")
        df = df.reset_index(drop=True)
        df["row_id"] = df.index

        def is_coherent(text):
            if pd.isna(text):
                return False

            text = str(text).strip()

            if len(text.split()) < 20:
                return False

            words = text.split()
            n = len(words)

            # 1. Too many repeated symbol-only tokens, e.g. "* * *" or "- - -"
            symbol_tokens = sum(1 for w in words if re.fullmatch(r"[\*\-\_]+", w))
            if n > 0 and symbol_tokens / n > 0.3:
                return False

            # 2. Too many corrupted / replacement-character tokens
            # catches outputs like "�:// �:// �://"
            bad_char_tokens = sum(1 for w in words if "�" in w)
            if n > 0 and bad_char_tokens / n > 0.1:
                return False

            # 3. Single-token repetition ratio
            # e.g. "heron heron heron..." or "heres heres heres..."
            cleaned_words = [
                re.sub(r"[^a-zA-Z0-9']+", "", w.lower())
                for w in words
            ]
            cleaned_words = [w for w in cleaned_words if w]

            if len(cleaned_words) >= 20:
                most_common_ratio = max(
                    cleaned_words.count(w) for w in set(cleaned_words)
                ) / len(cleaned_words)

                if most_common_ratio > 0.25:
                    return False

            # 4. Repeated n-gram detection
            # catches "it's been a while" repeated, or "it's about the student's" repeated
            def repeated_ngram_ratio(tokens, ngram_size):
                if len(tokens) < ngram_size * 3:
                    return 0

                ngrams = [
                    tuple(tokens[i:i + ngram_size])
                    for i in range(len(tokens) - ngram_size + 1)
                ]

                if not ngrams:
                    return 0

                most_common = max(ngrams.count(g) for g in set(ngrams))
                return most_common / len(ngrams)

            for ngram_size in [2, 3, 4, 5]:
                if repeated_ngram_ratio(cleaned_words, ngram_size) > 0.18:
                    return False

            # 5. Too many very short fragment tokens
            # catches things like "-US -US -US" or "no - no - no"
            short_tokens = sum(1 for w in cleaned_words if len(w) <= 2)
            if len(cleaned_words) > 0 and short_tokens / len(cleaned_words) > 0.45:
                return False

            # 6. Keep a small list of known weird markers
            # This is not the main logic, just extra safety.
            lower = text.lower()
            weird_markers = [
                "self-revol",
                "please monday",
                "using:",
                "commments"
            ]
            if any(m in lower for m in weird_markers):
                return False

            return True

        df["coherent"] = df["generated_text"].apply(is_coherent)

        df["resource_assumption_score"] = np.nan

        df["length"] = df["generated_text"].astype(str).apply(lambda x: len(x.split()))

        # only mark when coherent == True 
        df.to_parquet(
            f"./report/results/{country_code}_{label}_{loop}_intervention_scored_template.parquet",
            index=False
        )

        # Export only coherent rows that need scoring
        to_score = df[
            (df["coherent"] == True) &
            (df["resource_assumption_score"].isna())
        ].copy()

        to_score_cols = [
            "row_id",
            "country",
            "model",
            "k",
            "alpha",
            "task",
            "generated_text"
        ]

        to_score[to_score_cols].to_json(
            f"./report/results/{country_code}_{label}_{loop}_to_score.jsonl",
            orient="records",
            lines=True,
            force_ascii=False
        )

        print(f"./report/results/{country_code}_{label}_{loop}_to_score.jsonl has been created")

        """


        # Step 2: Mark resource assumption scores, and merge the new jsonl back to parquet
        """
        Prompts:
        Please assign a resource_assumption_score to each item below using the following 1–7 scale:
        1 = strongly assumes limited socioeconomic resources
        2 = moderately assumes limited socioeconomic resources
        3 = slightly assumes limited socioeconomic resources
        4 = socioeconomically neutral / no clear assumption
        5 = slightly assumes abundant socioeconomic resources
        6 = moderately assumes abundant socioeconomic resources
        7 = strongly assumes abundant socioeconomic resources

        Return JSONL only. Each line should contain row_id and resource_assumption_score. If an item is incoherent or not valid educational advice, return resource_assumption_score as null and coherent as false.
        
        template_path = f"./report/results/{country_code}_{label}_{loop}_intervention_scored_template.parquet"
        scores_path = f"./report/results/{country_code}_{label}_{loop}_scores_all.jsonl"
        out_path = f"./report/results/{country_code}_{label}_{loop}_intervention_scored.parquet"

        df = pd.read_parquet(template_path)
        scores = pd.read_json(scores_path, lines=True)

        df["row_id"] = df["row_id"].astype(int)
        scores["row_id"] = scores["row_id"].astype(int)

        # Update coherent if I mark some rows as incoherent
        if "coherent" in scores.columns:
            coherent_map = scores.dropna(subset=["coherent"]).set_index("row_id")["coherent"]
            df.loc[df["row_id"].isin(coherent_map.index), "coherent"] = (
                df.loc[df["row_id"].isin(coherent_map.index), "row_id"].map(coherent_map)
            )

        # Merge score
        score_map = scores.set_index("row_id")["resource_assumption_score"]

        df.loc[df["row_id"].isin(score_map.index), "resource_assumption_score"] = (
            df.loc[df["row_id"].isin(score_map.index), "row_id"].map(score_map)
        )

        # Update rows explicitly marked as incoherent in scores
        if "coherent" in scores.columns:
            bad_ids = scores.loc[
                scores["coherent"].astype(str).str.lower().isin(["false", "0", "0.0"]),
                "row_id"
            ]

            df.loc[df["row_id"].isin(bad_ids), "coherent"] = False

        # Incoherent rows should not have scores
        df["resource_assumption_score"] = pd.to_numeric(
            df["resource_assumption_score"],
            errors="coerce"
        )

        df["coherent"] = df["coherent"].apply(
            lambda x: False if str(x).lower() in ["false", "0", "0.0", "nan", "none"] else True
        ).astype(bool)

        df.loc[df["coherent"] == False, "resource_assumption_score"] = np.nan
        df.to_parquet(out_path, index=False)

        print("Saved:", out_path)
        print("Total rows:", len(df))
        print("Coherent rows:", df["coherent"].sum())
        print("Scored rows:", df["resource_assumption_score"].notna().sum())
        """
        

        # Graphs generation start
        # Step 3: Draw steering graphs
        """
        df_int = pd.read_parquet(f'./report/results/{country_code}_{label}_{loop}_intervention_scored.parquet')

        df_int = df_int.loc[df_int["coherent"] == True].copy()
        df_int = df_int.loc[pd.notnull(df_int["resource_assumption_score"])].copy()
        df_int = df_int.loc[df_int["alpha"].isin([-10, -5, 0, 5, 10])].copy()

        # Graph: ESCS intervention effect on resource assumptions
        plt.rcParams.update({'font.size': 18})

        plt.figure(figsize=(6,6))
        df_int['model2'] = df_int['model'].replace({
            'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral-7B-Instruct',
            'meta-llama/Llama-2-7b-chat-hf': 'Llama-2-7B-Chat',
            'lmsys/vicuna-7b-v1.5': 'Vicuna-7B'
        })
        df_int = df_int.sort_values('model2')
        sns.set_palette("dark")
        sns.pointplot(
            x='alpha',
            y='resource_assumption_score',
            hue='model2',
            data=df_int,
            ci=95,
            alpha=0.8,
            dodge=0.2,
            errwidth=2,
            capsize=0.1
        )
        # Remove title
        plt.title('')

        # Axis labels
        # Resource assumption scores indicate the degree to which the advice assumes limited or abundant socioeconomic resources.
        plt.ylabel('SE resource assumption score')
        plt.xlabel('Intervention (alpha)')

        # Remove plot spines (the borders)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.ylim(0.8, 7.2)

        # Remove legend boundary
        plt.legend(title='', fontsize='medium', frameon=False)
        plt.tight_layout()
        plt.savefig(
            f"./report/figures/{country_code}_intervention_score_by_alpha_{loop}.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


        # Graph: Coherence rate by intervention alpha
        df_coh = pd.read_parquet(
            f'./report/results/{country_code}_{label}_{loop}_intervention_scored.parquet'
        )

        df_coh['model2'] = df_coh['model'].replace({
            'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral-7B-Instruct',
            'meta-llama/Llama-2-7b-chat-hf': 'Llama-2-7B-Chat',
            'lmsys/vicuna-7b-v1.5': 'Vicuna-7B'
        })

        # Use all alpha values if you want to show collapse boundary.
        # If you only want the same range as the score figure, use [-10, -5, 0, 5, 10].
        df_coh = df_coh.loc[df_coh["alpha"].isin([-15, -10, -5, 0, 5, 10, 15])].copy()

        df_coh["coherent_numeric"] = df_coh["coherent"].astype(bool).astype(int)

        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(6, 6))

        sns.set_palette("dark")
        sns.pointplot(
            x='alpha',
            y='coherent_numeric',
            hue='model2',
            data=df_coh,
            ci=95,
            alpha=0.8,
            dodge=0.2,
            errwidth=2,
            capsize=0.1
        )

        plt.title('')
        plt.ylabel('Coherence rate')
        plt.xlabel('Intervention (alpha)')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)

        plt.ylim(-0.05, 1.05)

        plt.legend(title='', fontsize='medium', frameon=False)
        plt.tight_layout()
        plt.savefig(
            f"./report/figures/{country_code}_intervention_coherence_by_alpha_{loop}.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        """

        loop += 1