import os
import sys
import numpy as np
from encoder_only_lstm import LSTM_1Classifier, LSTM_2Classifier, LSTM_1_MultiFactorial, LSTM_2_MultiFactorial
from finetuning import FineTuning, MultifactorialFineTuning
from load_data import combine_dataframes, prepare_dataset
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import keras

class ModelTrainingConsole:
    def __init__(self):
        """
        Initialize the training console with common parameters.
        
        Sets up:
            - BERT model path
            - Maximum sequence length
            - Base paths for model storage
            - Data paths for humor and non-humor datasets
            - Paths for model evaluation and testing results
        """
        self.bert_path = 'bert-large-cased'
        self.max_length = 64
        self.base_paths = {
            'binary': 'models/binary',
            'multifactorial': 'models/multifactorial',
            'testing': {
                'binary': 'models/binary',
                'multifactorial': 'models/multifactorial'
            },
            'evaluation': {
                'binary': 'models/binary',
                'multifactorial': 'models/multifactorial'
            }
        }
        self.data_paths = {
            'humor': "data/classification/complete_dataset_chistes.csv",
            'no_humor': "data/classification/data_with_no_humour.csv",
            'humor_captioning_1': "data/classification/captionning_blip.csv",
            'humor_captioning_2': "data/classification/captionning_vlt.csv",
        }

    def evaluate_and_save_test_results(self, model, test_dataset, model_name, classifier_type, file_name =None, sample_size:int = 0):
        """
        Evaluate model on test set and save results with F1 score, accuracy, and recall.
        
        Args:
            model: The trained model (BERT or LSTM)
            test_dataset: Dataset to evaluate the model on
            model_name: Name of the model for saving results (e.g. 'lstm1', 'bert')
            classifier_type: Type of classifier ('binary' or 'multifactorial')
            file_name: Optional name for the output file
            sample_size: Size of training sample used, for tracking purposes
        
        Returns:
            dict: Dictionary containing evaluation metrics (accuracy, f1, recall)
            
        Saves:
            - Test evaluation results to a text file including:
                - Accuracy, F1 score, and recall metrics
                - Debug information about predictions
                - Class distribution information
                - Number of samples
        """
        
        try:
            if isinstance(model, (FineTuning, MultifactorialFineTuning)):
                # For BERT models, get metrics directly from trainer
                metrics = model.trainer.evaluate(test_dataset)
                
                # Create testing results directory if it doesn't exist
                eval_dir = os.path.join(self.base_paths['testing'][classifier_type], 'testing')
                os.makedirs(eval_dir, exist_ok=True)
                
                if file_name == None:
                    output_file = os.path.join(eval_dir, f"{model_name}_{classifier_type}_test_evaluation.txt")
                else:
                    output_file = os.path.join(eval_dir, f"{model_name}_{classifier_type}_{file_name}_{sample_size}_test_evaluation.txt")
                with open(output_file, "w") as f:
                    f.write(f"Test Set Evaluation Results for {model_name} ({classifier_type}):\n")
                    f.write("-" * 50 + "\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                
                print(f"\nTest evaluation results saved to {output_file}")
                return metrics
            else:
                # Get predictions and true labels
                test_inputs = model.encode_data(test_dataset)
                predictions_raw = model.predict(test_inputs[0], test_inputs[1], test_inputs[2])
                true_labels = test_inputs[3]
                
                # Convert predictions to binary values for binary classification
                if classifier_type == 'binary':
                    predictions = (predictions_raw > 0.5).astype(int)
                else:  # For multifactorial case
                    predictions = np.argmax(predictions_raw, axis=1)
                    true_labels = np.argmax(true_labels, axis=1)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(true_labels, predictions),
                    'f1': f1_score(true_labels, predictions, 
                                average='binary' if classifier_type == 'binary' else 'weighted'),
                    'recall': recall_score(true_labels, predictions, 
                                        average='binary' if classifier_type == 'binary' else 'weighted')
                }
                
                # Create testing results directory if it doesn't exist
                eval_dir = os.path.join(self.base_paths['testing'][classifier_type], 'testing')
                os.makedirs(eval_dir, exist_ok=True)
                # Save test results to file
                if file_name == None:
                    output_file = os.path.join(eval_dir, f"{model_name}_{classifier_type}_test_evaluation.txt")
                else:
                    output_file = os.path.join(eval_dir, f"{model_name}_{classifier_type}_{file_name}_{sample_size}_test_evaluation.txt")
                with open(output_file, "w") as f:
                    f.write(f"Test Set Evaluation Results for {model_name} ({classifier_type}):\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                    f.write(f"Recall: {metrics['recall']:.4f}\n")
                    
                    # Add some additional debug information
                    f.write("\nDebug Information:\n")
                    f.write(f"Unique predicted classes: {np.unique(predictions)}\n")
                    f.write(f"Unique true labels: {np.unique(true_labels)}\n")
                    f.write(f"Number of samples: {len(predictions)}\n")
                    
                    if classifier_type == 'binary':
                        # Add class distribution
                        f.write("\nClass Distribution:\n")
                        f.write(f"Predicted positives: {np.sum(predictions == 1)}\n")
                        f.write(f"Predicted negatives: {np.sum(predictions == 0)}\n")
                        f.write(f"Actual positives: {np.sum(true_labels == 1)}\n")
                        f.write(f"Actual negatives: {np.sum(true_labels == 0)}\n")
                
                print(f"\nTest evaluation results saved to {output_file}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                
                
                return metrics
        except Exception as e:
            print(f"Error on evaluate_and_save_test_results as: {e}")
    
    def save_validation_results(self, model, validation_dataset, model_name, classifier_type, file_name =None, sample_size:int = 0):
        """
        Save final validation results with complete distribution information.
        
        Args:
            model: The trained model (BERT or LSTM)
            validation_dataset: Dataset to validate the model on
            model_name: Name of the model (e.g. 'lstm1', 'bert')
            classifier_type: Type of classifier ('binary' or 'multifactorial')
            file_name: Optional name for the output file
            sample_size: Size of training sample used, for tracking purposes
        
        Returns:
            dict: Dictionary containing validation metrics
            
        Saves:
            - Validation results to a text file including:
                - Accuracy, F1 score, and recall metrics
                - Detailed class distribution information
                - Debug information about predictions
                - Sample counts and unique class information
        """
        try:
            if isinstance(model, (FineTuning, MultifactorialFineTuning)):
                metrics = model.trainer.evaluate()
                
                if file_name == None: 
                    eval_dir = os.path.join(self.base_paths['evaluation'][classifier_type], 'evaluation')
                    os.makedirs(eval_dir, exist_ok=True)
                
                    output_file = os.path.join(eval_dir, f"{model_name}_{file_name}_{sample_size}_validation_results.txt")
                else:
                    eval_dir = os.path.join(self.base_paths['evaluation'][classifier_type], 'evaluation')
                    os.makedirs(eval_dir, exist_ok=True)
                    output_file = os.path.join(eval_dir, f"{model_name}_{file_name}_{sample_size}_validation_results.txt")
                    
                with open(output_file, "w") as f:
                    f.write(f"Final Validation Results for {model_name} ({classifier_type}):\n")
                    f.write("-" * 50 + "\n")
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                
                print(f"\nValidation results saved to {output_file}")
                return metrics
            else:
                # Get predictions and true labels
                val_inputs = model.encode_data(validation_dataset)
                predictions_raw = model.predict(val_inputs[0], val_inputs[1], val_inputs[2])
                true_labels = val_inputs[3]
                
                # Ensure numpy arrays and handle array depth
                predictions_raw = np.array(predictions_raw)
                true_labels = np.array(true_labels)
                
                # Convert predictions to binary values for binary classification
                if classifier_type == 'binary':
                    predictions = (predictions_raw.squeeze() > 0.5).astype(int)
                    if len(true_labels.shape) > 1:
                        true_labels = true_labels.squeeze()
                else:  # For multifactorial case
                    predictions = np.argmax(predictions_raw, axis=1)
                    true_labels = np.argmax(true_labels, axis=1)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'f1': f1_score(true_labels, predictions, 
                            average='binary' if classifier_type == 'binary' else 'weighted'),
                'recall': recall_score(true_labels, predictions, 
                                    average='binary' if classifier_type == 'binary' else 'weighted')
            }
            
            # Create evaluation directory based on classifier type
            eval_dir = os.path.join(self.base_paths['evaluation'][classifier_type], 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save validation results
            output_file = os.path.join(eval_dir, f"{model_name}_validation_results.txt")
            with open(output_file, "w") as f:
                f.write(f"Final Validation Results for {model_name} ({classifier_type}):\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                
                # Add detailed information
                f.write("\nDebug Information:\n")
                f.write(f"Number of validation samples: {len(predictions)}\n")
                f.write(f"Unique predicted classes: {np.unique(predictions)}\n")
                f.write(f"Unique true labels: {np.unique(true_labels)}\n")
                
                # Add class distribution
                f.write("\nClass Distribution:\n")
                if classifier_type == 'binary':
                    f.write(f"Predicted positives: {np.sum(predictions == 1)}\n")
                    f.write(f"Predicted negatives: {np.sum(predictions == 0)}\n")
                    f.write(f"Actual positives: {np.sum(true_labels == 1)}\n")
                    f.write(f"Actual negatives: {np.sum(true_labels == 0)}\n")
                else:
                    f.write("Predicted class counts:\n")
                    for i, count in enumerate(np.bincount(predictions)):
                        f.write(f"Class {i}: {count}\n")
                    f.write("\nTrue class counts:\n")
                    for i, count in enumerate(np.bincount(true_labels)):
                        f.write(f"Class {i}: {count}\n")
            
            print(f"\nValidation results saved to {output_file}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print("\nClass distribution:")
            if classifier_type == 'binary':
                print(f"Predicted - Positives: {np.sum(predictions == 1)}, Negatives: {np.sum(predictions == 0)}")
                print(f"Actual - Positives: {np.sum(true_labels == 1)}, Negatives: {np.sum(true_labels == 0)}")
            
            return metrics
            
        except Exception as e:
            print(f"Error in save_validation_results: {str(e)}")
            print("Shapes:")
            print(f"predictions_raw shape: {predictions_raw.shape}")
            print(f"true_labels shape: {true_labels.shape}")
            if 'predictions' in locals():
                print(f"processed predictions shape: {predictions.shape}")
            raise

    def train_lstm1_binary(self, epochs = 10 , batch_size = 16):
        """
        Train and evaluate LSTM Model 1 for binary classification.
        
        Args:
            epochs (int): Number of training epochs (default: 10)
            batch_size (int): Size of training batches (default: 16)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Sets random seeds for reproducibility
            2. Prepares dataset from humor and non-humor sources
            3. Builds and compiles LSTM Model 1
            4. Trains model with early stopping
            5. Saves model to binary classification directory
            6. Evaluates on validation and test sets
        """
        print("\n=== Training LSTM Model 1 ===")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("\nPreparing dataset...")
        dataset = combine_dataframes(self.data_paths['humor'], self.data_paths['no_humor'])
        splits = prepare_dataset(dataset)
        
        classifier = LSTM_1Classifier(self.bert_path, self.max_length)
        model = classifier.build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("\nModel architecture created. Starting training...")
        
        train_dataset = splits['train']
        validation_dataset = splits['validation']
        
        history = classifier.train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=3
        )
        
        save_path = os.path.join(self.base_paths['binary'], 'lstm1_model.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path,save_format="h5")
        print(f"Model saved to {save_path}")
        
        validation_metrics = self.save_validation_results(
        classifier, validation_dataset, 'lstm1', 'binary'
    )
        
        test_metrics = self.evaluate_and_save_test_results(
            classifier, splits['test'], 'lstm1', 'binary'
        )
        return test_metrics

    def train_lstm2_binary(self, epochs:int = 10, batch_size:int = 16):
        """
        Train and evaluate LSTM Model 2 for binary classification.
        
        Args:
            epochs (int): Number of training epochs (default: 10)
            batch_size (int): Size of training batches (default: 16)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Sets random seeds for reproducibility
            2. Prepares dataset from humor and non-humor sources
            3. Builds and compiles LSTM Model 2 with modified architecture
            4. Trains model with early stopping
            5. Saves model to binary classification directory
            6. Evaluates on validation and test sets
        """
        print("\n=== Training LSTM Model 2 ===")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("\nPreparing dataset...")
        dataset = combine_dataframes(self.data_paths['humor'], self.data_paths['no_humor'])
        splits = prepare_dataset(dataset)
        
        classifier = LSTM_2Classifier(self.bert_path, self.max_length)
        model = classifier.build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("\nModel architecture created. Starting training...")
        
        train_dataset = splits['train']
        validation_dataset = splits['validation']
        
        
        history = classifier.train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=3
        )
        
        # Save model in binary directory
        save_path = os.path.join(self.base_paths['binary'], 'lstm2_model.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path,save_format="h5")
        print(f"Model saved to {save_path}")
        
        
        validation_metrics = self.save_validation_results(
        classifier, validation_dataset, 'lstm2', 'binary'
        )
        
        test_metrics = self.evaluate_and_save_test_results(
            classifier, splits['test'], 'lstm2', 'binary'
        )
        return test_metrics

    def train_finetuned_bert(self, sample_size=500, num_epochs: int = 100, all_data: bool = False, captioning:bool = False):
        """
        Train and evaluate fine-tuned BERT model for binary classification.
        
        Args:
            sample_size (int): Number of training examples to use (default: 500)
            num_epochs (int): Number of training epochs (default: 100)
            all_data (bool): Whether to use entire dataset instead of sample (default: False)
            captioning (bool): Whether to use captioning dataset (default: False)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Prepares dataset (either standard or captioning based on parameters)
            2. Samples data or uses full dataset based on all_data parameter
            3. Initializes and trains BERT model
            4. Saves model with appropriate naming convention
            5. Evaluates on validation and test sets
        """
        print("\n=== Training Fine-tuned BERT Model ===")
        
        print("\nPreparing dataset...")
        if captioning:
            dataset = combine_dataframes(
                self.data_paths['humor_captioning_1'],
                self.data_paths['no_humor']
            )
        else:
            dataset = combine_dataframes(
                self.data_paths['humor'],
                self.data_paths['no_humor']
            )
        
        # Validate dataset is not empty
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Please check input data files.")
            
        splits = prepare_dataset(dataset)
        
        if all_data:
            sample_size = splits['train'].num_rows
            train_data = splits['train']
        else:
            train_data = splits['train'].shuffle(seed=42).select(range(sample_size))
            
        val_data = splits['validation']
        test_data = splits['test']
        
        finetuning = FineTuning(model_name=self.bert_path)
        
        output_dir = self.base_paths['binary']
        os.makedirs(output_dir, exist_ok=True)
        
        training_output = finetuning.train(
            train_data,
            val_data,
            output=output_dir,
            num_epochs=num_epochs,
            
            
        )
        if captioning:
            filename = self.data_paths['humor_captioning_1'].replace("data/classification/","").replace(".csv","")
        else:
            filename = self.data_paths['humor'].replace("data/classification/","").replace(".csv","")
        save_path = os.path.join(output_dir, f'{filename}_finetuning_binary_model_{sample_size}_{num_epochs}')
        finetuning.save_model(save_path)
        
        validation_metrics = self.save_validation_results(
            finetuning, val_data, 'finetuning_bert', 'binary',
            file_name=filename, sample_size=sample_size
        )
        
        test_metrics = self.evaluate_and_save_test_results(
            finetuning, test_data, 'finetuning_bert', 'binary',
            file_name=filename, sample_size=sample_size
        )
        return test_metrics

    def train_lstm1_multifactorial(self,epochs:int = 10, batch_size:int = 16):
        """
        Train and evaluate LSTM Model 1 for multi-class classification.
        
        Args:
            epochs (int): Number of training epochs (default: 10)
            batch_size (int): Size of training batches (default: 16)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Sets random seeds for reproducibility
            2. Prepares dataset with one-hot encoded labels
            3. Builds and compiles LSTM Model 1 for multi-class classification
            4. Trains model with early stopping
            5. Saves model to multifactorial directory
            6. Evaluates on validation and test sets
        """
        print("\n=== Training LSTM Model 1 Multifactorial ===")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("\nPreparing dataset...")
        dataset, label_encoder = combine_dataframes(
            self.data_paths['humor'], 
            self.data_paths['no_humor'],
            one_hot=True
        )
        splits = prepare_dataset(dataset, column="nivel_risa_encoded")
        
        classifier = LSTM_1_MultiFactorial(self.bert_path, self.max_length, multifactorial=True)
        model = classifier.build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        train_dataset = splits['train']
        validation_dataset = splits['validation']
        
        print("\nModel architecture created. Starting training...")
        history = classifier.train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=3
        )
        
        # Save model in multifactorial directory
        save_path = os.path.join(self.base_paths['multifactorial'], 'lstm1_model.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path, save_format='h5')
        print(f"Model saved to {save_path}")
        
        validation_metrics = self.save_validation_results(
        classifier, validation_dataset, 'lstm1', 'multifactorial'
        )
        
        test_metrics = self.evaluate_and_save_test_results(
            classifier, splits['test'], 'lstm1', 'multifactorial'
        )
        return test_metrics

    def train_lstm2_multifactorial(self, epochs:int = 10, batch_size:int = 16):
        """
        Train and evaluate LSTM Model 2 for multi-class classification.
        
        Args:
            epochs (int): Number of training epochs (default: 10)
            batch_size (int): Size of training batches (default: 16)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Sets random seeds for reproducibility
            2. Prepares dataset with one-hot encoded labels
            3. Builds and compiles LSTM Model 2 for multi-class classification
            4. Trains model with early stopping
            5. Saves model to multifactorial directory
            6. Evaluates on validation and test sets
        """

        print("\n=== Training LSTM Model 2 Multifactorial ===")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("\nPreparing dataset...")
        dataset, label_encoder = combine_dataframes(
            self.data_paths['humor'], 
            self.data_paths['no_humor'],
            one_hot=True
        )
        splits = prepare_dataset(dataset, column="nivel_risa_encoded")
        
        classifier = LSTM_2_MultiFactorial(self.bert_path, self.max_length, multifactorial=True)
        model = classifier.build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        train_dataset = splits['train']
        validation_dataset = splits['validation']
        
        print("\nModel architecture created. Starting training...")
        history = classifier.train(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=3
        )
        
        # Save model in multifactorial directory
        save_path = os.path.join(self.base_paths['multifactorial'], 'lstm2_model.h5')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path,save_format="h5")
        print(f"Model saved to {save_path}")
        
        validation_metrics = self.save_validation_results(
        classifier, validation_dataset, 'lstm2', 'multifactorial'
        )
        
        test_metrics = self.evaluate_and_save_test_results(
            classifier, splits['test'], 'lstm2', 'multifactorial'
        )
        return test_metrics

    def train_multifactorial_finetuning(self, sample_size:int =7000, num_epochs: int = 100, all_data: bool = False, captioning: bool = False):
        """
        Train and evaluate fine-tuned BERT model for multi-class classification.
        
        Args:
            sample_size (int): Number of training examples to use (default: 7000)
            num_epochs (int): Number of training epochs (default: 100)
            all_data (bool): Whether to use entire dataset instead of sample (default: False)
            captioning (bool): Whether to use captioning dataset (default: False)
        
        Returns:
            dict: Test metrics after training
            
        Process:
            1. Prepares dataset with one-hot encoded labels
            2. Configures model for appropriate number of output classes
            3. Trains on either sampled or full dataset based on parameters
            4. Saves model with appropriate naming convention
            5. Evaluates on validation and test sets
        """
        if captioning:
            dataset, label_encoder = combine_dataframes(
                self.data_paths['humor_captioning_2'],
                self.data_paths['no_humor'],
                one_hot=True
            )
        else:
            dataset, label_encoder = combine_dataframes(
                self.data_paths['humor'],
                self.data_paths['no_humor'],
                one_hot=True
            )
            
        splits = prepare_dataset(dataset, column='nivel_risa_encoded')
        
        num_labels = len(label_encoder.classes_)
        print(f"Number of nivel_risa categories: {num_labels}")
        
        model = MultifactorialFineTuning(num_labels=num_labels, label_encoder=label_encoder)
        if all_data:
            train_sample = splits['train']
            sample_size = splits['train'].num_rows
        else:
            train_sample = splits['train'].shuffle(seed=42).select(range(sample_size))
        validation_sample = splits['validation']
        test_sample = splits['test']
        
        # Use multifactorial directory for output
        output_dir = self.base_paths['multifactorial']
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting training...")
        training_results = model.train(
            train_dataset=train_sample,
            validation_dataset=validation_sample,
            output=output_dir,
            num_epochs=num_epochs
        )
        
        if captioning:
            filename = self.data_paths['humor_captioning_2'].replace("data/classification/","").replace(".csv","")
        else:
            filename = self.data_paths['humor'].replace("data/classification/","").replace(".csv","")
        
        # Save in multifactorial directory
        save_path = os.path.join(output_dir, f'{filename}_finetuning_multifactorial_model_{sample_size}_{num_epochs}')
        model.save_model(save_path)
        
        print(f"Saved model on {save_path}")
        
        
        validation_metrics = self.save_validation_results(
        model, validation_sample, 'finetuning_bert', 'multifactorial',file_name=filename,sample_size=sample_size
        )
        
        test_metrics = self.evaluate_and_save_test_results(
            model, test_sample, 'finetuning_bert', 'multifactorial',file_name=filename,sample_size=sample_size
        )
        return test_metrics
    
    def run_selected_training(self):
        """
        Execute training for any combination of selected models.
        
        Available models:
        1. LSTM Model 1 (Binary)
        2. LSTM Model 2 (Binary)
        3. LSTM Model 1 (Multifactorial)
        4. LSTM Model 2 (Multifactorial)
        5. BERT Binary
        6. BERT Multifactorial
        7. BERT Binary with Captioning
        8. BERT Multifactorial with Captioning
        
        Process:
            1. Prompts for common training parameters
            2. Displays available models
            3. Allows user to select multiple models
            4. Trains only the selected models with specified parameters
        """
        # Get common training parameters
        lstm_epochs = int(input("Enter number of epochs for LSTM training (default: 10): ") or 10)
        lstm_batch_size = int(input("Enter batch size for LSTM training (default: 8): ") or 8)
        bert_epochs = int(input("Enter number of epochs for BERT models (default: 100): ") or 100)
        
        # Get sample sizes for BERT models if needed
        bert_binary_size = None
        bert_multi_size = None
        use_all_data = str(input("Use all data for BERT models? (y/n, default: n): ") or 'n').lower() == 'y'
        
        if not use_all_data:
            bert_binary_size = int(input("Enter sample size for BERT binary models (default: 5000): ") or 5000)
            bert_multi_size = int(input("Enter sample size for BERT multifactorial models (default: 5000): ") or 5000)
        
        print("\nAvailable models for training:")
        print("1. LSTM Model 1 (Binary)")
        print("2. LSTM Model 2 (Binary)")
        print("3. LSTM Model 1 (Multifactorial)")
        print("4. LSTM Model 2 (Multifactorial)")
        print("5. BERT Binary")
        print("6. BERT Multifactorial")
        print("7. BERT Binary with Captioning")
        print("8. BERT Multifactorial with Captioning")
        
        # Get user input for model selection
        model_selection = input("\nEnter the numbers of models to train (comma-separated, e.g., '1,3,4,6'): ")
        selected_models = [int(x.strip()) for x in model_selection.split(',')]
        
        print("\n=== Beginning Selected Models Training ===")
        
        # Dictionary mapping model numbers to their training functions with parameters
        training_functions = {
            1: lambda: self.train_lstm1_binary(lstm_epochs, lstm_batch_size),
            2: lambda: self.train_lstm2_binary(lstm_epochs, lstm_batch_size),
            3: lambda: self.train_lstm1_multifactorial(lstm_epochs, lstm_batch_size),
            4: lambda: self.train_lstm2_multifactorial(lstm_epochs, lstm_batch_size),
            5: lambda: self.train_finetuned_bert(
                sample_size=bert_binary_size if not use_all_data else None,
                num_epochs=bert_epochs,
                all_data=use_all_data
            ),
            6: lambda: self.train_multifactorial_finetuning(
                sample_size=bert_multi_size if not use_all_data else None,
                num_epochs=bert_epochs,
                all_data=use_all_data
            ),
            7: lambda: self.train_finetuned_bert(
                sample_size=bert_binary_size if not use_all_data else None,
                num_epochs=bert_epochs,
                all_data=use_all_data,
                captioning=True
            ),
            8: lambda: self.train_multifactorial_finetuning(
                sample_size=bert_multi_size if not use_all_data else None,
                num_epochs=bert_epochs,
                all_data=use_all_data,
                captioning=True
            )
        }
        
        # Train only selected models
        for model_num in selected_models:
            if model_num in training_functions:
                print(f"\nTraining Model {model_num}...")
                try:
                    training_functions[model_num]()
                except Exception as e:
                    print(f"Error training model {model_num}: {str(e)}")
                    continue
            else:
                print(f"Warning: Model {model_num} is not a valid selection. Skipping.")


def main():
    """
    Main entry point for the model training console.
    
    Provides an interactive command-line interface for:
        1. Training any combination of models
        2. Configuring training parameters
        3. Exiting the program
    """
    console = ModelTrainingConsole()
    
    while True:
        print("\n=== Humor Classification Model Training Console ===")
        print("1. Train Selected Models")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        try:
            if choice == '1':
                console.run_selected_training()
            elif choice == '2':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")
                
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your data paths and try again.")

if __name__ == "__main__":
    main()