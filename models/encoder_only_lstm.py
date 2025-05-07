

from transformers import TFBertModel
import keras
import numpy as np

class BaseClassifier:
    def __init__(self, bert_path='bert-large-cased', max_length=64, multifactorial= False):
        """
        Base class for BERT-based classification models.
        
        Provides common functionality for model initialization, data encoding,
        training, prediction, and evaluation. Serves as parent class for specific
        classifier implementations.
        
        Attributes:
            bert_path (str): Path to pre-trained BERT model
            max_length (int): Maximum sequence length for input texts
            model: The Keras model instance
            multifactorial (bool): Whether model handles multiple classes
        """
        self.bert_path = bert_path
        self.max_length = max_length
        self.model = None
        self.multifactorial = multifactorial
        
    def save_model(self, filepath):
        """Save model in H5 format"""
        if self.model is not None:
            self.model.save(filepath, save_format='h5')
        else:
            raise ValueError("Model hasn't been built yet")
        
    class BertLayer(keras.layers.Layer):
        """
        Custom Keras layer wrapping BERT model.
        
        Creates a non-trainable BERT layer that outputs the last hidden states
        of the model for use in downstream tasks.
        
        Args:
            bert_path (str): Path to pre-trained BERT model
            **kwargs: Additional arguments passed to parent Layer class
        """
        def __init__(self, bert_path='bert-large-cased', **kwargs):
            super().__init__(**kwargs)
            # Specify output_hidden_states=True to get all hidden states
            self.bert_path = bert_path
            self.bert = TFBertModel.from_pretrained(bert_path, output_hidden_states=True)
            self.bert.trainable = False
            
        def call(self, inputs):
            input_ids, attention_mask, token_type_ids = inputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,                
            )
            # Use the last hidden state instead of pooler output
            return outputs.last_hidden_state
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "bert_path": self.bert_path
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    def build_model(self):
        """
        Abstract method to define model architecture.
        
        Must be implemented by subclasses to create their specific model architectures.
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement build_model")
    
    def encode_data(self, dataset):
        """
        Encode input data for model processing.
        
        Args:
            dataset: Dataset containing input_ids, attention_mask, token_type_ids, and labels
            
        Returns:
            tuple: For binary classification:
                - input_ids (np.array)
                - attention_mask (np.array)
                - token_type_ids (np.array)
                - labels (np.array)
            For multifactorial:
                - Same as above but labels are one-hot encoded for 6 categories
        """
        if self.multifactorial:
            input_ids = np.array([item["input_ids"] for item in dataset])
            attention_mask = np.array([item["attention_mask"] for item in dataset])
            token_type_ids = np.array([item["token_type_ids"] for item in dataset])
            # Get labels and convert to one-hot encoding
            labels = np.array([item["nivel_risa_encoded"] for item in dataset])
            # Convert to one-hot encoding for 6 categories (0-5)
            one_hot_labels = np.eye(6)[labels]
            return input_ids, attention_mask, token_type_ids, one_hot_labels
        
        else:
            input_ids = np.array([item["input_ids"] for item in dataset])
            attention_mask = np.array([item["attention_mask"] for item in dataset])
            token_type_ids = np.array([item["token_type_ids"] for item in dataset])
            labels = np.array([item["labels"] for item in dataset])
            return input_ids, attention_mask, token_type_ids, labels
    
    def train(self, train_dataset, validation_dataset, epochs=10, batch_size=8, patience=3):
        """
        Train the model on provided datasets.
        
        Args:
            train_dataset: Training data
            validation_dataset: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
            
        Returns:
            keras.callbacks.History: Training history
        """
        train_inputs = self.encode_data(train_dataset)
        val_inputs = self.encode_data(validation_dataset)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, mode='min'
        )
        history = self.model.fit(
            x=[train_inputs[0], train_inputs[1], train_inputs[2]],
            y=train_inputs[3],
            validation_data=( [val_inputs[0], val_inputs[1], val_inputs[2]], val_inputs[3] ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        return history
    
    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        Make predictions using the trained model.
        
        Args:
            input_ids: BERT input token IDs
            attention_mask: BERT attention mask
            token_type_ids: BERT token type IDs
            
        Returns:
            np.array: Model predictions
        """
        return self.model.predict([input_ids, attention_mask, token_type_ids])
    
    def evaluate(self, test_dataset):
        """
        Evaluate model performance on test dataset.
        
        Args:
            test_dataset: Test data to evaluate on
            
        Returns:
            dict: Evaluation metrics (loss and accuracy)
        """
        test_inputs = self.encode_data(test_dataset)
        results = self.model.evaluate(
            [test_inputs[0], test_inputs[1], test_inputs[2]],
            test_inputs[3]
        )
        return dict(zip(self.model.metrics_names, results))


class LSTM_1Classifier(BaseClassifier):
    """
    Binary classifier combining BERT with dual bidirectional LSTM layers.
    
    Architecture:
        1. BERT layer for text encoding
        2. Self-attention layer
        3. Concatenation of BERT output and attention
        4. First bidirectional LSTM with sequence return
        5. Dropout layer
        6. Second bidirectional LSTM
        7. Single unit sigmoid output for binary classification
    """
    
    def build_model(self):
        """
        Creates a binary classification model combining BERT with dual LSTM layers.
        
        Architecture:
            1. Input layers:
                - input_ids: Token indices of input sequence (shape: [batch_size, 64])
                - attention_mask: Mask for padding (shape: [batch_size, 64])
                - token_type_ids: Segment tokens (shape: [batch_size, 64])
            
            2. BERT Layer:
                - Non-trainable BERT-large-cased model
                - Outputs last hidden states
            
            3. Attention mechanism:
                - Self-attention on BERT outputs
                - Helps focus on relevant parts of input
            
            4. Feature combination:
                - Concatenates BERT outputs with attention outputs
                - Enriches representation with attention information
            
            5. First LSTM layer:
                - Bidirectional LSTM with 256 units
                - Returns sequences for hierarchical processing
                - Total 512 features (256 * 2 directions)
            
            6. Regularization:
                - Dropout layer with 0.3 rate
                - Prevents overfitting
            
            7. Second LSTM layer:
                - Bidirectional LSTM with 256 units
                - Returns final sequence state
                - Total 512 features (256 * 2 directions)
            
            8. Output:
                - Dense layer with 1 unit
                - Sigmoid activation for binary classification
        
        Returns:
            keras.Model: Compiled model with inputs [input_ids, attention_mask, token_type_ids]
                        and binary classification output
        """
        input_ids = keras.layers.Input(shape=(64,), dtype='int32', name='input_ids')
        attention_mask = keras.layers.Input(shape=(64,), dtype='int32', name='attention_mask')
        token_type_ids = keras.layers.Input(shape=(64,), dtype='int32', name='token_type_ids')
        
        bert_layer = self.BertLayer(self.bert_path)
        sequence_output = bert_layer([input_ids, attention_mask, token_type_ids])
        
        attention = keras.layers.Attention()([sequence_output, sequence_output])
        
        merge_layer = keras.layers.Concatenate()([sequence_output, attention])
        
        LSTM1_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256, return_sequences=True)
        )(merge_layer)
        
        dropout_layer = keras.layers.Dropout(rate=0.3)(LSTM1_layer)
        
        LSTM2_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256)
        )(dropout_layer)
        
        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(LSTM2_layer)
        
        self.model = keras.Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output_layer
        )
        return self.model


class LSTM_2Classifier(BaseClassifier):
    """
    Binary classifier combining BERT with single bidirectional LSTM layer.
    
    Architecture:
        1. BERT layer for text encoding
        2. Self-attention layer
        3. Concatenation of BERT output and attention
        4. Single bidirectional LSTM
        5. Single unit sigmoid output for binary classification
    """
    
    def build_model(self):
        """
        Creates a binary classification model combining BERT with single LSTM layer.
        
        Architecture:
            1. Input layers:
                - input_ids: Token indices of input sequence (shape: [batch_size, 64])
                - attention_mask: Mask for padding (shape: [batch_size, 64])
                - token_type_ids: Segment tokens (shape: [batch_size, 64])
            
            2. BERT Layer:
                - Non-trainable BERT-large-cased model
                - Outputs last hidden states
            
            3. Attention mechanism:
                - Self-attention on BERT outputs
                - Helps focus on relevant parts of input
            
            4. Feature combination:
                - Concatenates BERT outputs with attention outputs
                - Enriches representation with attention information
            
            5. LSTM layer:
                - Single Bidirectional LSTM with 256 units
                - Returns final sequence state
                - Total 512 features (256 * 2 directions)
            
            6. Output:
                - Dense layer with 1 unit
                - Sigmoid activation for binary classification
        
        Returns:
            keras.Model: Compiled model with inputs [input_ids, attention_mask, token_type_ids]
                        and binary classification output
        """
        input_ids = keras.layers.Input(shape=(64,), dtype='int32', name='input_ids')
        attention_mask = keras.layers.Input(shape=(64,), dtype='int32', name='attention_mask')
        token_type_ids = keras.layers.Input(shape=(64,), dtype='int32', name='token_type_ids')
        
        bert_layer = self.BertLayer(self.bert_path)
        sequence_output = bert_layer([input_ids, attention_mask, token_type_ids])
        
        attention = keras.layers.Attention()([sequence_output, sequence_output])
        
        merge_layer = keras.layers.Concatenate()([sequence_output, attention])
        
        LSTM1_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256)
        )(merge_layer)
            
        output_layer = keras.layers.Dense(units=1, activation='sigmoid')(LSTM1_layer)
        
        self.model = keras.Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output_layer
        )
        return self.model
    
class LSTM_2_MultiFactorial(BaseClassifier):
    """
    Multi-class classifier combining BERT with single bidirectional LSTM layer.
    
    Architecture:
        1. BERT layer for text encoding
        2. Self-attention layer
        3. Concatenation of BERT output and attention
        4. Single bidirectional LSTM
        5. Six unit softmax output for multi-class classification
    """
    
    def build_model(self):
        """
        Creates a multi-class classification model combining BERT with single LSTM layer.
        
        Architecture:
            1. Input layers:
                - input_ids: Token indices of input sequence (shape: [batch_size, 64])
                - attention_mask: Mask for padding (shape: [batch_size, 64])
                - token_type_ids: Segment tokens (shape: [batch_size, 64])
            
            2. BERT Layer:
                - Non-trainable BERT-large-cased model
                - Outputs last hidden states
            
            3. Attention mechanism:
                - Self-attention on BERT outputs
                - Helps focus on relevant parts of input
            
            4. Feature combination:
                - Concatenates BERT outputs with attention outputs
                - Enriches representation with attention information
            
            5. LSTM layer:
                - Single Bidirectional LSTM with 256 units
                - Returns final sequence state
                - Total 512 features (256 * 2 directions)
            
            6. Output:
                - Dense layer with 6 units (one per class)
                - Softmax activation for multi-class classification
        
        Returns:
            keras.Model: Compiled model with inputs [input_ids, attention_mask, token_type_ids]
                        and multi-class classification output (6 classes)
        """
        input_ids = keras.layers.Input(shape=(64,), dtype='int32', name='input_ids')
        attention_mask = keras.layers.Input(shape=(64,), dtype='int32', name='attention_mask')
        token_type_ids = keras.layers.Input(shape=(64,), dtype='int32', name='token_type_ids')
        
        bert_layer = self.BertLayer(self.bert_path)
        sequence_output = bert_layer([input_ids, attention_mask, token_type_ids])
        
        attention = keras.layers.Attention()([sequence_output, sequence_output])
        
        merge_layer = keras.layers.Concatenate()([sequence_output, attention])
        
        LSTM1_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256)
        )(merge_layer)
            
        output_layer = keras.layers.Dense(units=6, activation='softmax')(LSTM1_layer)
        
        self.model = keras.Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output_layer
        )
        return self.model
    
    
class LSTM_1_MultiFactorial(BaseClassifier):
    """
    Multi-class classifier combining BERT with dual bidirectional LSTM layers.
    
    Architecture:
        1. BERT layer for text encoding
        2. Self-attention layer
        3. Concatenation of BERT output and attention
        4. First bidirectional LSTM with sequence return
        5. Dropout layer
        6. Second bidirectional LSTM
        7. Six unit softmax output for multi-class classification
    """
    
    def build_model(self):
        """
        Creates a multi-class classification model combining BERT with dual LSTM layers.
        
        Architecture:
            1. Input layers:
                - input_ids: Token indices of input sequence (shape: [batch_size, 64])
                - attention_mask: Mask for padding (shape: [batch_size, 64])
                - token_type_ids: Segment tokens (shape: [batch_size, 64])
            
            2. BERT Layer:
                - Non-trainable BERT-large-cased model
                - Outputs last hidden states
            
            3. Attention mechanism:
                - Self-attention on BERT outputs
                - Helps focus on relevant parts of input
            
            4. Feature combination:
                - Concatenates BERT outputs with attention outputs
                - Enriches representation with attention information
            
            5. First LSTM layer:
                - Bidirectional LSTM with 256 units
                - Returns sequences for hierarchical processing
                - Total 512 features (256 * 2 directions)
            
            6. Regularization:
                - Dropout layer with 0.3 rate
                - Prevents overfitting
            
            7. Second LSTM layer:
                - Bidirectional LSTM with 256 units
                - Returns final sequence state
                - Total 512 features (256 * 2 directions)
            
            8. Output:
                - Dense layer with 6 units (one per class)
                - Softmax activation for multi-class classification
        
        Returns:
            keras.Model: Compiled model with inputs [input_ids, attention_mask, token_type_ids]
                        and multi-class classification output (6 classes)
        """
        input_ids = keras.layers.Input(shape=(64,), dtype='int32', name='input_ids')
        attention_mask = keras.layers.Input(shape=(64,), dtype='int32', name='attention_mask')
        token_type_ids = keras.layers.Input(shape=(64,), dtype='int32', name='token_type_ids')
        
        bert_layer = self.BertLayer(self.bert_path)
        sequence_output = bert_layer([input_ids, attention_mask, token_type_ids])
        
        attention = keras.layers.Attention()([sequence_output, sequence_output])
        
        merge_layer = keras.layers.Concatenate()([sequence_output, attention])
        
        LSTM1_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256, return_sequences=True)
        )(merge_layer)
        
        dropout_layer = keras.layers.Dropout(rate=0.3)(LSTM1_layer)
        
        LSTM2_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(units=256)
        )(dropout_layer)
        
        output_layer = keras.layers.Dense(units=6, activation='softmax')(LSTM2_layer)
        
        self.model = keras.Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=output_layer
        )
        return self.model