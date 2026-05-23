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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin

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


def classify_value(x, q25, q75, catagory):
    # hisei & homepos
    # quartile at 125 & 175
    if catagory == 'hisei':
        if x < q25:
            return "low"
        elif x > q75:
            return "high"
        else:
            return "medium"
    elif catagory == 'homepos':
        if x < q25:
            return "few"
        elif x > q75:
            return "many"
        else:
            return "some"
    elif catagory == 'paredint':
            # ---- PRIMARY / LOWER EDUCATION ----
            if x < 6:
                return "did not complete primary education"
            if x < 9:
                return "completed primary education"
            if x < 12:
                return "completed lower secondary education"

            # ---- UPPER SECONDARY ----
            if x < 13:
                return "completed upper secondary education"

            # ---- TERTIARY EDUCATION ----
            if x < 15:
                return "completed short-cycle tertiary education"
            if x == 15:
                return "completed bachelor-level tertiary education"

            # ---- BACHELOR / POSTGRADUATE ----
            return "completed bachelor or postgraduate education"


ISCED_MAP = {
    0: "less than primary education",
    1: "primary education",
    2: "lower secondary education",
    3: "upper secondary education",
    4: "post-secondary non-tertiary education",
    5: "short-cycle tertiary education",
    6: "Bachelor's degree or equivalent",
    7: "Master's degree or equivalent",
    8: "Doctoral degree or equivalent",
}

OCC_SPEC = {
    0: {"mode": "in", "phrase": "the armed forces"},
    1: {"mode": "as", "phrase": "a manager"},
    2: {"mode": "as", "phrase": "a professional"},
    3: {"mode": "as", "phrase": "a technician or associate professional"},
    4: {"mode": "as", "phrase": "a clerical support worker"},
    5: {"mode": "as", "phrase": "a service or sales worker"},
    6: {"mode": "as", "phrase": "a skilled agricultural, forestry or fishery worker"},
    7: {"mode": "as", "phrase": "a craft or related trades worker"},
    8: {"mode": "as", "phrase": "a plant or machine operator, or an assembler"},
    9: {"mode": "as", "phrase": "an elementary occupations worker"},
}

def isco08_sentence(subject: str, isco_code, unknown="unknown"):
    if pd.isna(isco_code):
        return f"{subject}'s occupation is {unknown}."

    try:
        code_str = str(int(isco_code))
    except (TypeError, ValueError):
        return f"{subject}'s occupation is {unknown}."

    if not code_str or not code_str[0].isdigit():
        return f"{subject}'s occupation is {unknown}."

    major = int(code_str[0])

    spec = OCC_SPEC.get(major)
    if spec is None:
        return f"{subject}'s occupation is {unknown}."

    if spec["mode"] == "in":
        return f"{subject} works in {spec['phrase']}."
    else:
        return f"{subject} works as {spec['phrase']}."

for country_code in country_list:
    loop = 5
    print(f'Start analyzing country {country_code}')
    print("Start reading sample file")
    df = pd.read_parquet(f'./data/2018/PISA_{country_code}_full.parquet')

    country_fullname = country_name_map[country_code]

    print(country_code, 'ESCS range:', df['ESCS'].min(), '->', df['ESCS'].max())
    print(country_code, 'HISEI range:', df['HISEI'].min(), '->', df['HISEI'].max())
    print(country_code, 'PAREDINT range:', df['PAREDINT'].min(), '->', df['PAREDINT'].max())
    print(country_code, 'HOMEPOS range:', df['HOMEPOS'].min(), '->', df['HOMEPOS'].max())
    print('-----------------------------')

    # ESCS, HISEI, PAREDINT, HOMEPOS
    # The PISA index of economic, social and cultural status (ESCS)
    # Parents’ highest occupational status (HISEI)
    # Parents’ highest education, in years (PAREDINT)
    # Home possessions (HOMEPOS)
    hisei_q25 = df["HISEI"].quantile(0.25)
    hisei_q75 = df["HISEI"].quantile(0.75)

    homepos_q25 = df["HOMEPOS"].quantile(0.25)
    homepos_q75 = df["HOMEPOS"].quantile(0.75)

    print(country_code, 'HISEI q25/q75:', hisei_q25, hisei_q75)
    print(country_code, 'HOMEPOS q25/q75:', homepos_q25, homepos_q75)

    while loop <= 6:
        # ==================== Prompting ====================
        print("Start prompting")
        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            tokenizer = llama.LlamaTokenizerFast.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
            
            # Generate prompts for ESCS simulation
            statements = []
            for index, row in df.iterrows():
                hisei = row["HISEI"]
                paredint = row["PAREDINT"]
                homepos = row["HOMEPOS"]
                parent_highest_edu = row["HISCED"]
                mother_occupation = row["OCOD1"]
                father_occupation = row["OCOD2"]

                parent_edu_text = ISCED_MAP.get(int(parent_highest_edu), "Unknown") if not pd.isna(parent_highest_edu) else "Unknown"

                # using all 3 attributes
                # if (loop == 0):
                #     statements.append(tokenizer(f'USER: Generate a statement by a middle school student in {country_fullname} with \
                #                                 HISEI as {hisei}, PAREDINT as {paredint}, HOMEPOS as {homepos} as stated by PISA.\n \
                #                                 ASSISTANT: In 2022, the student with HISEI as {hisei}, PAREDINT as {paredint}, HOMEPOS as {homepos} \
                #                                 said that', return_tensors="pt")['input_ids'])

                # # using HISEI and PAREDINT only
                # if (loop == 1):
                #     statements.append(tokenizer(f'USER: Generate a statement by a middle school student in {country_fullname} with \
                #                                 HISEI as {hisei}, PAREDINT as {paredint} as stated by PISA.\n \
                #                                 ASSISTANT: In 2022, the student with HISEI as {hisei}, PAREDINT as {paredint} \
                #                                 said that', return_tensors="pt")['input_ids'])
                    
                # # using HISEI and HOMEPOS only
                # if (loop == 2):
                #     statements.append(tokenizer(f'USER: Generate a statement by a middle school student in {country_fullname} with \
                #                                 HISEI as {hisei}, HOMEPOS as {homepos} as stated by PISA.\n \
                #                                 ASSISTANT: In 2022, the student with HISEI as {hisei}, HOMEPOS as {homepos} \
                #                                 said that', return_tensors="pt")['input_ids'])
                    
                # # using PAREDINT and HOMEPOS only
                # if (loop == 3):
                #     statements.append(tokenizer(f'USER: Generate a statement by a middle school student in {country_fullname} with \
                #                                 PAREDINT as {paredint}, HOMEPOS as {homepos} as stated by PISA.\n \
                #                                 ASSISTANT: In 2022, the student with PAREDINT as {paredint}, HOMEPOS as {homepos} \
                #                                 said that', return_tensors="pt")['input_ids'])
                    
                    
                # using all 3 attributes, but human readable language
                # if (loop == 4):
                #     hisei_level = classify_value(hisei, 28.48, 73.91)
                #     homepos_level = classify_value(homepos, -1.1941, 0.3651)
                #     paredint_level = classify_value(paredint, 0, 0, 'paredint')

                #     statements.append(tokenizer(f'Write a statement from a middle school student in {country_fullname}. \
                #                                 Their family background is: {hisei_level} parental occupation, {homepos_level} home possessions, \
                #                                 and parents {paredint_level}.\n \
                #                                 ASSISTANT: In 2022, this student said that', return_tensors="pt")['input_ids'])
                    

                if (loop == 5):
                    statements.append(tokenizer(f'A middle school student from {country_fullname} has HISEI={hisei}, \
                                                PAREDINT={paredint}, HOMEPOS={homepos}. \n \
                                                In 2018, the student said that', return_tensors="pt")['input_ids'])

                if (loop == 6):
                    hisei_level = classify_value(hisei, hisei_q25, hisei_q75, 'hisei')
                    homepos_level = classify_value(homepos, homepos_q25, homepos_q75, 'homepos')
                    paredint_level = classify_value(paredint, 0, 0, 'paredint')
                    statements.append(tokenizer(f'A middle school student from {country_fullname} \
                                                comes from a family where the parents have {hisei_level}-prestige jobs, \
                                                own {homepos_level} possessions at home, and {paredint_level}. \n \
                                                In 2018, this student wrote:', return_tensors="pt")['input_ids'])


                # if (loop == 7):
                #     statements.append(tokenizer(f'USER: Generate a statement by a middle school student in {country_fullname} with \
                #                                 HISEI as {hisei}, PAREDINT as {paredint}, HOMEPOS as {homepos} as stated by PISA.\n \
                #                                 ASSISTANT: In 2018, the student with HISEI as {hisei}, PAREDINT as {paredint}, HOMEPOS as {homepos} \
                #                                 said that', return_tensors="pt")['input_ids'])
                    
                # if (loop == 8):
                #     mum = isco08_sentence("The student's mother", mother_occupation)
                #     dad = isco08_sentence("The student's father", father_occupation)

                #     prompt = (
                #         f"A middle school student from {country_fullname}. "
                #         f"{mum} "
                #         f"{dad} "
                #         f"Their highest education level is {parent_edu_text}. "
                #         f"They own {car_count} and {book_count} at home.\n"
                #         f"In 2018, this student wrote about their future job expectations:"
                #     )
                #     if index == 0:
                #         print(prompt)

                #     statements.append(tokenizer(prompt, return_tensors="pt")['input_ids'])


            pickle.dump(statements, open(f'./report/results/{country_code}_{model_name.replace("/", "_")}_{label}_{loop}.pkl', 'wb'))


        # ==================== Extracting Activations ====================
        print("Start extracting activations")

        # To reduce memory, grab the last token in the first loop
        def extract_attention_head_activations(model, statements):
            # read dim from model.config
            HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
            
            # get from config
            hidden_dim = model.config.hidden_size  # get hidden layer size
            num_heads = model.config.num_attention_heads
            head_dim = hidden_dim // num_heads
            
            num_samples = len(statements)
            num_layers = model.config.num_hidden_layers
            
            features = np.empty((num_samples, 1, num_layers, num_heads, head_dim), dtype=np.float16)
            
            for idx, prompt in enumerate(tqdm(statements)):
                with torch.no_grad():
                    with TraceDict(model, HEADS) as ret:
                        output = model(prompt.to(device), output_hidden_states=True, output_attentions=True)
                        
                        for layer_idx, head in enumerate(HEADS):
                            last_token = ret[head].output[0, -1, :].cpu().numpy()
                            features[idx, 0, layer_idx, :, :] = last_token.astype(np.float16).reshape(num_heads, head_dim)
                        
                del output
                del ret
                if idx % 10 == 0:
                    torch.cuda.empty_cache()
            return features


        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            model = llama.LlamaForCausalLM.from_pretrained(model_name, cache_dir='./model', low_cpu_mem_usage=True, torch_dtype=torch.float16, token=HUGGINGFACE_TOKEN).to(device)
            # Extract activations for label
            statements = pickle.load(open(f'./report/results/{country_code}_{model_name.replace("/", "_")}_{label}_{loop}.pkl', 'rb'))
            labels = np.array(df[label].astype(float))
            features = extract_attention_head_activations(model, statements)
            pickle.dump((features, labels), open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_features.pkl", 'wb'))


        # ==================== Probing ====================
        print("Start probing")

        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            features, labels = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_features.pkl", 'rb'))
            num_layers = features.shape[2]
            num_heads = features.shape[3]
            performance = np.zeros((num_layers, num_heads))
            ridge_dict = {}
            for i in tqdm(range(num_layers)):
                ridge_dict[i] = {}
                for j in range(num_heads):
                    kf = KFold(n_splits=2, shuffle=True, random_state=42)
                    for train_indices, test_indices in kf.split(range(features.shape[0])):
                        X_train = features[train_indices, 0, i, j, :]
                        X_test = features[test_indices, 0, i, j, :]
                        y_train = np.array(labels)[train_indices]
                        y_test = np.array(labels)[test_indices]
                        # set fit_intercept to true to learn a baseline value (an intercept), so predictions don’t have to 
                        # be forced through zero and can more accurately reflect the overall level of the target variable
                        ridge_model = Ridge(alpha=1, fit_intercept=True)
                        ridge_model.fit(X_train, y_train)
                        ridge_dict[i][j] = ridge_model
                        y_pred = ridge_model.predict(X_test)
                        performance[i, j] += spearmanr(y_test, y_pred).statistic
            performance /= 2
            pickle.dump(performance, open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_performance.pkl", 'wb'))
            pickle.dump(ridge_dict, open(f"./report/results/{country_code}_{model_name.replace('/', '_')}_{label}_{loop}_ridge.pkl", "wb"))


        # ==================== Intervention ====================
        def lt_modulated_vector_add(head_output, layer_name):
            layer_index = layer_name[len('model.layers.'):]
            layer_index = int(layer_index[:layer_index.index('.')])
            head_output = rearrange(head_output.detach().cpu(), 'b s (h d) -> b s h d', h=model.config.num_attention_heads)
            for head_index in head_dict[layer_index]:
                head_output[:, -1, head_index, :] += alpha  * focal_ridge_dict[(layer_index, head_index)] * np.std(features[:, 0, layer_index, head_index, :], axis=0)
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output.to('cuda')

        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            tokenizer = llama.LlamaTokenizer.from_pretrained(model_name, cache_dir='./model', token=HUGGINGFACE_TOKEN)
            class LlamaForCausalLMWithGenerate(llama.LlamaForCausalLM, GenerationMixin):
                pass
            model = LlamaForCausalLMWithGenerate.from_pretrained(model_name, cache_dir='./model', low_cpu_mem_usage=True, torch_dtype=torch.float16, 
                                                        token=HUGGINGFACE_TOKEN).to('cuda:0')
            model.eval()
            model.config.use_cache = False
            performance = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_performance.pkl", 'rb'))
            features, labels = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_features.pkl", 'rb'))
            trained_ridge_dict = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_ridge.pkl", 'rb'))
            topics = [
                'reading comprehension',
                'mathematics homework',
                'university preparation',
                'science learning',
                'art learning',
                'exam preparation',
                'career planning',
                'class participation',
                'independent study'
            ]
            results = []
            # for k in reversed([16, 32, 48, 64, 80, 96]):
            for k in reversed([16]):
                top_indices = np.dstack(
                    np.unravel_index(
                        np.argsort(performance.ravel()),
                        (32, 32)
                    )
                )[0][-k:, :][::-1]

                norms = []

                for layer, head in top_indices:
                    layer = int(layer)
                    head = int(head)

                    coef = trained_ridge_dict[layer][head].coef_
                    std = np.std(features[:, 0, layer, head, :], axis=0)
                    vec = coef * std

                    norms.append(np.linalg.norm(vec))

                print("Intervention vector norms")
                print("country:", country_code, "model:", model_name, "k:", k)
                print("mean norm:", np.mean(norms))
                print("max norm:", np.max(norms))
                print("min norm:", np.min(norms))

                for alpha in tqdm(reversed([-15, -10, -5, 0, 5, 10, 15])):
                    for topic in topics:
                        print(k, alpha, topic)
                        original_prompt = (f"A middle school student in {country_fullname} is struggling with {topic}. "
                                           f"Write supportive personalised learning advice for this student.")
                        top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), (32, 32)))[0][-k:, :][::-1]
                        focal_ridge_dict = {}
                        for i in top_indices:
                            ridge_model = trained_ridge_dict[i[0]][i[1]]
                            focal_ridge_dict[tuple(i)] = ridge_model.coef_
                        head_dict = {}
                        for i in top_indices:
                            if i[0] not in head_dict:
                                head_dict[i[0]] = [i[1]]
                            else:
                                head_dict[i[0]].append(i[1])
                        with TraceDict(model, [f'model.layers.{i}.self_attn.head_out' for i in sorted(list(set(top_indices[:,0])))], edit_output=lt_modulated_vector_add) as ret: 
                            input_ids = tokenizer(f"USER: {original_prompt}\nASSISTANT:", return_tensors="pt")['input_ids']
                            model_gen_tokens = model.generate(
                                input_ids.to(device),
                                max_new_tokens=150,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                use_cache=False
                            )

                        prompt_len = input_ids.shape[-1]
                        new_tokens = model_gen_tokens[0][prompt_len:]

                        model_gen_str = tokenizer.decode(
                            new_tokens,
                            skip_special_tokens=True
                        ).strip()
                        results.append([country_fullname, model_name, k, alpha, topic, model_gen_str])
            pickle.dump(results, open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_intervention_results.pkl", 'wb'))
        
        intervention_raw_results = []
        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            intervention_raw_results += pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_intervention_results.pkl", 'rb'))

        df_intervention_raw = pd.DataFrame(intervention_raw_results, columns=['country', 'model', 'k', 'alpha', 'task', 'generated_text'])
        output_path = f"./report/results/{country_code}_{label}_{loop}_intervention_raw.parquet"
        df_intervention_raw.to_parquet(output_path)
        print(f"parquet file {output_path} is generated.")


        # draw graphs
        # activated layers

        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            performance = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_performance.pkl", 'rb'))
            print(performance.max())
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(6, 6))
            norm = mcolors.Normalize(vmin=0, vmax=.9)
            plt.imshow(np.sort(performance[::-1, :]*(-1), axis=1)*(-1), cmap='YlGnBu', norm=norm)
            plt.grid(False)
            plt.xlabel("Head (Sorted)")
            plt.ylabel("Layer")
            plt.xticks([])
            plt.yticks(range(0,32,2), [i for i in range(32,0,-2)])
            cbar = plt.colorbar(orientation='vertical', fraction=0.0459, pad=0.04)
            cbar.outline.set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.tight_layout()
            plt.savefig(
                f"./report/figures/{country_code}_{model_name.replace('/','_')}_activated_layers_{loop}.png",
                dpi=300
            )
            
            print("Top accuracy heads")
            top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), (32, 32)))[0][-20:, :][::-1]
            print(top_indices)
            print("Top accuracy")
            print(performance[top_indices[:, 0], top_indices[:, 1]])

        # relationship between actual and predict label
        cmap = mcolors.LinearSegmentedColormap.from_list("red_white_blue", ["blue", "white", "red"])

        k = 32
        for model_name in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', 'lmsys/vicuna-7b-v1.5']:
            features, labels = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_features.pkl", 'rb'))
            performance = pickle.load(open(f"./report/results/{country_code}_{model_name.replace('/','_')}_{label}_{loop}_performance.pkl", 'rb'))
            top_indices = np.dstack(np.unravel_index(np.argsort(performance.ravel()), (32, 32)))[0][-k:, :]
            kf = KFold(n_splits=2, shuffle=True, random_state=42)
            ensemble_pred = np.zeros(labels.shape)
            for train_indices, test_indices in kf.split(range(features.shape[0])):
                for i, j in top_indices:
                    X_train = features[train_indices, 0, i, j, :]
                    X_test = features[test_indices, 0, i, j, :]
                    y_train = np.array(labels)[train_indices]
                    y_test = np.array(labels)[test_indices]
                    # set fit_intercept to true to learn a baseline value (an intercept), so predictions don’t have to 
                    # be forced through zero and can more accurately reflect the overall level of the target variable
                    ridge_model = Ridge(alpha=1, fit_intercept=True)
                    ridge_model.fit(X_train, y_train)
                    y_pred = ridge_model.predict(X_test)
                    ensemble_pred[test_indices] += y_pred
            ensemble_pred = ensemble_pred / k
            print(model_name, k, spearmanr(labels, ensemble_pred).statistic)
            plt.rcParams.update({'font.size': 14})
            plt.figure(figsize=(8.5, 6))
            plt.grid(False)
            plt.title(f"")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.scatter(ensemble_pred, labels, c = ensemble_pred, cmap=cmap, alpha=0.8, s=50, edgecolor='black')
            all_min = min(labels.min(), ensemble_pred.min())
            all_max = max(labels.max(), ensemble_pred.max())
            margin = 0.05 * (all_max - all_min)
            plt.xlim(all_min - margin, all_max + margin)
            plt.ylim(all_min - margin, all_max + margin)
            ax = plt.gca()
            ax.plot(
                [all_min - margin, all_max + margin],
                [all_min - margin, all_max + margin],
                linestyle='--',
                linewidth=1,
                color='gray',
                alpha=0.7
            )
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()

            # get mean/median for actual and predit label
            x_mean = float(np.mean(ensemble_pred))
            x_median = float(np.median(ensemble_pred))
            y_mean = float(np.mean(labels))
            y_median = float(np.median(labels))

            rho = spearmanr(labels, ensemble_pred).statistic

            # legend
            stats_text = (
                f"Spearman = {rho:.3f}\n"
                f"mean_predict = {x_mean:.3f}\n"
                f"mean_actual = {y_mean:.3f}\n"
                f"median_predict = {x_median:.3f}\n"
                f"median_actual = {y_median:.3f}"
            )
            ax.text(
                0.98, 0.02, stats_text,
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none'),
                clip_on=True
            )

            plt.tight_layout()
            plt.savefig(
                f"./report/figures/{country_code}_{model_name.replace('/','_')}_correlation_{loop}.png",
                dpi=300,
                bbox_inches='tight'
            )

        loop += 1


