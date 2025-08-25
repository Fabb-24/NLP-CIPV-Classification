import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch


class CustomModel:
    """
    A class to fine-tune a Bert-based model for classification tasks.
    """

    def __init__(self, model_name: str, train_dataset, val_dataset, test_dataset, labels: list, output_dir: str,
                 batch_size: int = 16, learning_rate: float = 5e-5, num_epochs: int = 3, weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0, seed: int = 42, es_patience: int = 4, es_threshold: float = 0.005):
        """
        Initializes the CustomModel.

        Args:
            model_name (str): Name of the pre-trained model to fine-tune.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
            labels (list): List of labels for classification.
            output_dir (str): Directory to save the model and tokenizer.
            batch_size (int): Batch size for training and evaluation.
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs for training.
            weight_decay (float): Weight decay for the optimizer.
            max_grad_norm (float): Maximum gradient norm for clipping.
            seed (int): Random seed for reproducibility.
            es_patience (int): Early stopping patience.
            es_threshold (float): Early stopping threshold.
        """

        print(f"[INFO] Initializing CustomModel with:\n"
              f"\tmodel name: {model_name},\n"
              f"\toutput dir: {output_dir},\n"
              f"\tbatch size: {batch_size},\n"
              f"\tlearning rate: {learning_rate},\n"
              f"\tnum epochs: {num_epochs},\n"
              f"\tweight decay: {weight_decay},\n"
              f"\tmax grad norm: {max_grad_norm},\n"
              f"\tseed: {seed}\n"
              f"\tearly stopping patience: {es_patience},\n"
              f"\tearly stopping threshold: {es_threshold}\n")

        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.labels = labels

        # Create label mappings
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for idx, label in enumerate(labels)}

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # Add new tokens to the tokenizer
        self.tokenizer.add_tokens(["[SPEAKER1]", "[SPEAKER2]", "[TURN_SEP]"])
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Define TrainingArguments
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            warmup_ratio=0.1,
            weight_decay=weight_decay,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            learning_rate=learning_rate,
            greater_is_better=False,
            save_total_limit=3,
            seed=seed,
            max_grad_norm=max_grad_norm,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last= True,
            optim="adamw_torch",
            adam_beta1= 0.9,
            adam_beta2= 0.999,
            adam_epsilon= 1e-8,
        )

        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=es_patience, early_stopping_threshold=es_threshold)]
        )

        print(f"[INFO] CustomModel initialized with:\n"
              f"\t{len(train_dataset)} train examples,\n"
              f"\t{len(val_dataset)} validation examples, and\n"
              f"\t{len(self.test_dataset)} test examples.\n\n")
        

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for model evaluation.

        Args:
            eval_pred: The evaluation predictions.

        Returns:
            A dictionary containing the computed metrics.
        """

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        # Use accuracy, precision, recall, and F1-score as metrics
        accuracy = report["accuracy"]
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        macro_f1 = report["macro avg"]["f1-score"]
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "macro_f1": macro_f1}


    def train(self):
        """
        Starts the training process.
        """

        print(f"[INFO] Starting training...")
        self.trainer.train()
        print(f"[INFO] Training complete.\n\n")


    def evaluate(self):
        """
        Evaluates the model on the test dataset and prints the classification report and confusion matrix.
        """

        # Print details of best model
        print(f"[INFO] Evaluation metrics:")
        metrics = self.trainer.evaluate()
        print(f"   eval_loss: {metrics['eval_loss']:.4f}, "
              f"\n   eval_accuracy: {metrics['eval_accuracy']:.4f}, "
              f"\n   eval_precision: {metrics['eval_precision']:.4f}, "
              f"\n   eval_recall: {metrics['eval_recall']:.4f}, "
              f"\n   eval_macro_f1: {metrics['eval_macro_f1']:.4f}")
        print("\n\n")

        predictions = self.trainer.predict(self.test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        # Print classification report for test dataset
        print(f"[INFO] Evaluation results on test dataset:")
        print(classification_report(true_labels, preds, target_names=self.labels, zero_division=0, digits=4))

        # Plot confusion matrix for test dataset
        print(f"[INFO] Plotting confusion matrix...")
        cm = confusion_matrix(true_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.title('Confusion Matrix on Test Dataset')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Train/Eval loss plot
        print(f"[INFO] Plotting training and evaluation loss...")
        history = self.trainer.state.log_history
        train_loss = []
        eval_loss = []
        epochs = []

        for entry in history:
            if 'loss' in entry and 'epoch' in entry:
                train_loss.append(entry['loss'])
                epochs.append(entry['epoch'])
            if 'eval_loss' in entry:
                eval_loss.append(entry['eval_loss'])

        minimum = min(len(train_loss), len(eval_loss), len(epochs))
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[:minimum], train_loss[:minimum], label='Training Loss', marker='o')
        plt.plot(epochs[:minimum], eval_loss[:minimum], label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def save_model(self):
        """
        Saves the trained model and tokenizer to the output directory.
        """

        print(f"[INFO] Saving model and tokenizer")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.trainer.save_state()
        print(f"[INFO] Model and tokenizer saved to {self.output_dir}.\n\n")