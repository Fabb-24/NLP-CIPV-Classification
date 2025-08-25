import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ConversationDataset(Dataset):
    """
    A dataset class for loading and processing conversation texts and their labels.
    """

    def __init__(self, texts, labels, tokenizer: PreTrainedTokenizer, max_length: int = 512, add_token_type_ids: bool = False):
        """
        Initializes the ConversationDataset.

        Args:
            texts (list): List of text samples.
            labels (list): List of corresponding labels.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding text data.
            max_length (int): Maximum length of tokenized sequences.
            add_token_type_ids (bool): Whether to include token type IDs in the output.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_token_type_ids = add_token_type_ids


    def get_texts(self):
        """
        Returns the texts in the dataset.
        """

        return self.texts
    

    def get_labels(self):
        """
        Returns the labels in the dataset.
        """

        return self.labels

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.texts)
    

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input and label.
        """

        if idx < 0 or idx >= len(self.texts):
            raise IndexError("Index out of bounds for dataset.")
        
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True,
        )

        # Create a dictionary to hold the tokenized input and label
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # Add token type ids if required
        if self.add_token_type_ids and 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return item