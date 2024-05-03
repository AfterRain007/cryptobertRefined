import pandas as pd
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import build_compute_metrics_fn
from ray.tune.schedulers import PopulationBasedTraining
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
import random
import torch
import os

def importAugmentedData():
    langCodeGT = ['it', 'fr', 'sv', 'da', 'pt',
                  'id', 'pl', 'hr', 'bg', 'fi',
                  'no', 'ru', 'es', 'nl', 'af',
                  'de', 'sk', 'cs', 'lv', 'sq']
    
    dfGT20 = pd.DataFrame()
    for lang in langCodeGT:
        temp = pd.read_csv(f'./augmented_data/dfTrain-{lang}GT.csv')
        dfGT20 = pd.concat([dfGT20, temp])
    
    dfGT10 = dfGT20[:int(len(dfGT20)/2)].copy()
    
    langCodeHNLP = ['zh', 'es', 'ru', 'jap', 
                    'de', 'fr', 'it', 'id']
    
    dfHNLP = pd.DataFrame()
    for lang in langCodeHNLP:
        temp = pd.read_csv(f'./augmented_data/dfTrain-{lang}HNLP.csv')
        dfHNLP = pd.concat([dfHNLP, temp])

    dfTrain = pd.read_csv("./data/dfTrain.csv")
    
    dfGT20 = pd.concat([dfTrain, dfGT20])
    dfGT10 = pd.concat([dfTrain, dfGT10])
    dfHNLP = pd.concat([dfTrain, dfHNLP])
    
    return dfGT20, dfGT10, dfHNLP

def prepData(dfTrain, dfVal, tokenizer, tokenSize = 128):
    class dataSet(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dfTrain['sen'] = dfTrain['sen'] + 1
    dfVal['sen'] = dfVal['sen'] + 1

    train_encodings = tokenizer(dfTrain['text'].tolist(), max_length = tokenSize, truncation=True, padding=True)
    val_encodings = tokenizer(dfVal['text'].tolist(), max_length = tokenSize, truncation=True, padding=True)

    train_dataset = dataSet(train_encodings, dfTrain['sen'].tolist())
    val_dataset = dataSet(val_encodings, dfVal['sen'].tolist())

    return train_dataset, val_dataset

def initialize(dfTrain, dfVal, modelName):
    f = open("./data/cryptoVocab.txt", "r")
    crypto_vocabulary = f.read().split(',')
    crypto_vocabulary = [term.replace('"', '') for term in crypto_vocabulary]

    if (modelName == "cardiffnlp/twitter-roberta-base-sentiment-latest") | (modelName == "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        tokenSize = 256
    else:
        tokenSize = 128

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    try:
        new_tokens = set(crypto_vocabulary) - set(tokenizer.vocab.keys())
    except:
        new_tokens = set(crypto_vocabulary) - set(tokenizer.get_vocab().keys())

    tokenizer.add_tokens(list(new_tokens))

    dfTrain, dfVal = prepData(dfTrain, dfVal, tokenizer, tokenSize)

    return dfTrain, dfVal, tokenizer

def get_model(modelName):
    model = AutoModelForSequenceClassification.from_pretrained(modelName)
    model.resize_token_embeddings(len(tokenizer))
    return model

def train_fn(config):
    checkpoint: train.Checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

def start(modelName, train_dataset, val_dataset):
    task_name = 'rte'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,  # config
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3,
        max_steps=-1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=0,  # config
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
        )

    tune_config = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "warmup_steps" : tune.choice([0, 250, 500]),
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )

    destination_directory = './ray_results'
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)
    ray.shutdown()
    torch.cuda.empty_cache()

    trainer = Trainer(
        model_init=get_model(modelName),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics_fn(task_name),
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        keep_checkpoints_num=1,
        n_trials=9,
        scheduler=scheduler,
        resources_per_trial={"cpu" : 2, "gpu" : 1},
        progress_reporter=reporter,
        stop={"training_iteration": 1},
        local_dir="./ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    ) 