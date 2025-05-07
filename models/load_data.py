import sklearn.metrics
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score
import sklearn
def combine_dataframes(file_humour, file_no_humour, one_hot:bool = False):
    """
    Combine humor and non-humor datasets into a single dataset.
    
    Args:
        file_humour (str): Path to CSV file containing humor data
        file_no_humour (str): Path to CSV file containing non-humor data
        
    Returns:
        Dataset: Combined dataset with humor and non-humor examples
    """
    try:
        if one_hot:
            df_no_humor = pd.read_csv(file_no_humour)
            df_no_humor.rename(columns={"label":"nivel_risa"},inplace=True)
            
            df_humor = pd.read_csv(file_humour)
            df_humor.rename(columns={"Chistes": "text"}, inplace=True)
            df_humor.drop(columns=["id_chiste"], inplace=True)
            
            df_combined = pd.concat([df_humor, df_no_humor], ignore_index=True)
            # Create label encoder for nivel_risa
            label_encoder = LabelEncoder()
            df_combined['nivel_risa_encoded'] = label_encoder.fit_transform(df_combined['nivel_risa'])
            
            # Store the original mapping
            label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            print("Label mapping:", label_mapping)
            
            dataset = Dataset.from_pandas(df_combined)
            return dataset, label_encoder
        else:
            df_no_humor = pd.read_csv(file_no_humour)
            df_no_humor["nivel_risa"] = 0
            
            df_humor = pd.read_csv(file_humour)
            df_humor.rename(columns={"Chistes": "text"}, inplace=True)
            df_humor['label'] = 1
            df_humor.drop(columns=["id_chiste"], inplace=True)
            
            df_combined = pd.concat([df_humor, df_no_humor], ignore_index=True)
            return Dataset.from_pandas(df_combined)
    except Exception as e:
        print(f"Error on combine_dataframes: {e}")
    
def tokenize_function(examples, model_name='bert-large-cased',column='label'):
    """
    Tokenize text examples using the BERT tokenizer.
    
    Args:
        examples (dict): Dictionary containing text examples
        
    Returns:
        dict: Tokenized examples with input_ids, attention_mask, and labels
    """
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name, unk_token="[UNK]")
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=64
        )
        tokenized['labels'] = examples[column]
        return tokenized
    except Exception as e:
        print(f"Error on tokenize_function: {e}")
    
def prepare_dataset(dataset,column="label"):
    """
    Prepare dataset by tokenizing and splitting into train/validation/test sets.
    
    Args:
        dataset (Dataset): Input dataset
        
    Returns:
        DatasetDict: Dictionary containing train, validation, and test splits
    """
    try:
        print("COLUMN NAME INSIDE PREPARE DATASET",column)
        encoded_dataset = dataset.map(
            lambda x: tokenize_function(x, column=column), 
            batched=True,
        )
        print("encoded_dataset INSIDE PREPARE DATASET",encoded_dataset)
        train_validation_test = encoded_dataset.train_test_split(test_size=0.3, seed=42)
        train_validation = train_validation_test['train'].train_test_split(test_size=0.1, seed=42)
        print("train_validation INSIDE PREPARE DATASET",train_validation)
        return DatasetDict({
            'train': train_validation['train'],
            'validation': train_validation['test'],
            'test': train_validation_test['test']
        })
    except Exception as e:
        print(f"Error on prepare_dataset: {e}")
    
    
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred (tuple): Tuple of predictions and labels
        
    Returns:
        dict: Dictionary containing accuracy metric
    """
    try:
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions,),
        "f1_score":sklearn.metrics.f1_score(labels,predictions,average='binary'),
        "f1_score_micro":sklearn.metrics.f1_score(labels,predictions,average='micro'),
        "recall":recall_score(labels,predictions)}
    except Exception as e:
            print(f"Error on compute_metrics: {e}")
        
def compute_metrics_multifactorial(encoder=None):
    """
    Compute evaluation metrics for model predictions.

    Args:
        binary (bool): Whether to compute metrics for binary classification
        encoder: LabelEncoder instance for non-binary classification
        
    Returns:
        dict: Dictionary containing metrics
    """
    def compute(eval_pred):
        
        try:
            
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            f1_score = sklearn.metrics.f1_score(labels,predictions,average='weighted')
            f1_score_micro = sklearn.metrics.f1_score(labels,predictions,average='micro')
            recall = recall_score(labels,predictions,average="weighted")
            pred_original = encoder.inverse_transform(predictions)
            labels_original = encoder.inverse_transform(labels)
            
            pred_examples = ','.join(map(str, pred_original[:5]))
            label_examples = ','.join(map(str, labels_original[:5]))
            
            return {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "f1_score_micro": f1_score_micro,
                "recall": recall,
                "predictions_sample": pred_examples,  
                "true_labels_sample": label_examples,
                "pred_mean": predictions.mean(),
                "pred_std": predictions.std(),
            }
                    
        except Exception as e:
                print(f"Error on compute_metrics_multifactorial: {e}")
    
    return compute
