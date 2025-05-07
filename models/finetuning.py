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
from models.load_data import compute_metrics_multifactorial, compute_metrics, combine_dataframes, tokenize_function, prepare_dataset
import argparse

class FineTuning:
    """
    A class for fine-tuning BERT models for text classification tasks.
    
    This class provides functionality for loading data, tokenizing text,
    training a BERT model, and making predictions.
    
    Attributes:
        model_name (str): Name of the pretrained BERT model
        num_labels (int): Number of classification labels
        max_length (int): Maximum sequence length for tokenization
        tokenizer: BERT tokenizer instance
        model: BERT model instance
    """
    
    def __init__(self, model_name='bert-large-cased', num_labels=2, max_length=64):
        """
        Initialize the FineTuning class.
        
        Args:
            model_name (str): Name of the pretrained BERT model
            num_labels (int): Number of classification labels
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name, unk_token="[UNK]")
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    
        
    def train(self, train_dataset, validation_dataset, 
              learning_rate=5e-5, batch_size=8, num_epochs=500, use_cpu=False, output="finetuning_binary_model"):
        """
        Train the BERT model on the provided datasets.
        
        Args:
            train_dataset (Dataset): Training dataset
            validation_dataset (Dataset): Validation dataset
            output_dir (str): Directory to save training outputs
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
            use_cpu (bool): Whether to use CPU instead of GPU
            
        Returns:
            TrainOutput: Training results
        """
        try:
            training_args = TrainingArguments(
                evaluation_strategy="steps",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                logging_steps=5,
                use_cpu=False,
                fp16=True,
                fp16_opt_level = "O1",
                save_total_limit=1,
                output_dir=output,
                load_best_model_at_end=True, 
                metric_for_best_model="eval_loss",  
                greater_is_better=False, 
            )
            
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping],
            )
            
            return self.trainer.train()
        except Exception as e:
            print(f"Error on train: {e}")
    
    def save_model(self, path='models/binary/bert-fine-tuned-humor'):
        """
        Save the fine-tuned model and tokenizer to disk.
        
        Args:
            path (str): Directory path to save the model.
        """
        try:
            # Usar os.makedirs con exist_ok=True para evitar errores si ya existe
            os.makedirs(path, exist_ok=True)

            # Guardar el modelo y el tokenizador
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            print(f"Model and tokenizer saved successfully at: {path}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
    
    def load_model(self, path='./bert-fine-tuned-humor'):
        """
        Load a fine-tuned model and tokenizer from disk.
        
        Args:
            path (str): Directory path containing the saved model
        """
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
    
    def evaluate(self, output_file="evaluation_results_finetuning_binary.txt"):
        """
        Evaluate the model on the validation dataset and save results to a text file.
        
        Args:
            output_file (str): Path to the output file where evaluation metrics will be saved.
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Realizar la evaluación
            metrics = self.trainer.evaluate()

            # Guardar los resultados en un archivo de texto
            with open(output_file, "w") as f:
                f.write("Evaluation Results:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Evaluation results saved to {output_file}")
            return metrics
        except Exception as e:
            print(f"An error occurred during evaluate: {e}")
            

    
    def predict(self, text):
        """
        Make predictions on new text input.
        
        Args:
            text (str): Input text for prediction
            
        Returns:
            tensor: Prediction probabilities for each class
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
            outputs = self.model(**inputs)
            return torch.nn.functional.softmax(outputs.logits, dim=1)
        except Exception as e:
            print(f"An error occurred during predict: {e}")


class MultifactorialFineTuning:
    """
    A class for fine-tuning BERT models for multiclass text classification tasks.

    This class provides functionality for loading data, tokenizing text,
    training a BERT model, and making predictions with multiple labels.

    Attributes:
        model_name (str): Name of the pretrained BERT model
        num_labels (int): Number of classification labels
        max_length (int): Maximum sequence length for tokenization
        tokenizer: BERT tokenizer instance
        model: BERT model instance
        label_encoder: Sklearn LabelEncoder instance for label transformation
    """

    def __init__(self, model_name='bert-large-cased', num_labels=None, max_length=64, label_encoder=None):
        """
        Initialize the MultifactorialFineTuning class.
        
        Args:
            model_name (str): Name of the pretrained BERT model
            num_labels (int): Number of classification labels
            max_length (int): Maximum sequence length for tokenization
            label_encoder: LabelEncoder instance for transforming labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.label_encoder = label_encoder
        self.tokenizer = BertTokenizer.from_pretrained(model_name, unk_token="[UNK]")
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    def train(self, train_dataset, validation_dataset, 
              learning_rate=5e-5, batch_size=8, num_epochs=100, use_cpu=False, 
              output="finetuning_multifactorial_model"):
        """
        Train the BERT model on the provided datasets.
        
        Args:
            train_dataset (Dataset): Training dataset
            validation_dataset (Dataset): Validation dataset
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
            use_cpu (bool): Whether to use CPU instead of GPU
            output (str): Directory to save training outputs
            
        Returns:
            TrainOutput: Training results
        """
        try:
            training_args = TrainingArguments(
                evaluation_strategy="steps",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                logging_steps=5,
                use_cpu=False,
                fp16=True,
                fp16_opt_level = "O1",
                save_total_limit=1,
                output_dir=output,
                load_best_model_at_end=True,  
                metric_for_best_model="eval_loss", 
                greater_is_better=False,
            )
            
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics_multifactorial(encoder=self.label_encoder),
                callbacks=[early_stopping],
            )
            
            return self.trainer.train()
        except Exception as e:
            print(f"Error during training: {e}")
    
    def save_model(self, path='models/multifactorial/finetuning_multifactorial_model'):
        """
        Save the fine-tuned model, tokenizer, and label encoder to disk.
        
        Args:
            path (str): Directory path to save the model
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save label encoder using pickle
            import pickle
            with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            print(f"Model, tokenizer, and label encoder saved successfully at: {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path='./bert-fine-tuned-humor'):
        """
        Load a fine-tuned model, tokenizer, and label encoder from disk.
        
        Args:
            path (str): Directory path containing the saved model
        """
        try:
            self.model = BertForSequenceClassification.from_pretrained(path)
            self.tokenizer = BertTokenizer.from_pretrained(path)
            
            # Load label encoder
            import pickle
            with open(os.path.join(path, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, text):
        """
        Make predictions on new text input.
        
        Args:
            text (str): Input text for prediction
            
        Returns:
            dict: Dictionary containing predicted label and probability distribution
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=self.max_length)
            outputs = self.model(**inputs)
            
            # Get probabilities for each class
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get the predicted class
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            
            # Convert back to original label
            original_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Create probability distribution dictionary
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
            
            return {
                'predicted_label': original_label,
                'probabilities': prob_dict
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    def evaluate(self, output_file="models/multifactorial/evaluation_results_multifactorial_finetuning.txt"):
        """
        Evaluate the model on the validation dataset and save results to a text file.
        
        Args:
            output_file (str): Path to the output file where evaluation metrics will be saved
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            metrics = self.trainer.evaluate()
            
            with open(output_file, "w") as f:
                f.write("Evaluation Results:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Evaluation results saved to {output_file}")
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            
 