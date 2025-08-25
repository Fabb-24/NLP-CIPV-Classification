import json
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .conversation_dataset import ConversationDataset


class TurnDataLoader(Dataset):
    """
    A data loader for processing conversation datasets into single-turn examples.
    """

    def __init__(self, json_dataset_path: str, tokenizer: PreTrainedTokenizer, test_size: float = 0.1, val_size: float = 0.2,
                 max_length: int = 512, random_state: int = 42, remove_stopwords: bool = True,
                 lemmatize: bool = True, add_token_type_ids: bool = False):
        """
        Initializes the TurnDataLoader.

        Args:
            json_dataset_path (str): Path to the JSON file containing conversation data.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text data.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training set to include in the validation split.
            max_length (int): Maximum length of input sequences.
            random_state (int): Random seed for reproducibility.
            remove_stopwords (bool): Whether to remove stopwords during tokenization.
            lemmatize (bool): Whether to lemmatize tokens during tokenization.
            add_token_type_ids (bool): Whether to add token type IDs to the input.
        """

        print(f"[INFO] Initializing TurnDataLoader with:\n"
              f"\tjson path: {json_dataset_path},\n"
              f"\ttest size: {test_size},\n"
              f"\tval size: {val_size},\n"
              f"\tmax length: {max_length},\n"
              f"\trandom state: {random_state},\n"
              f"\tremove stopwords: {remove_stopwords},\n"
              f"\tlemmatize: {lemmatize}\n")

        self.json_dataset_path = json_dataset_path
        self.tokenizer = tokenizer
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length
        self.random_state = random_state
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.add_token_type_ids = add_token_type_ids
        self.label_encoder = LabelEncoder()
        
        # Check if the JSON file already exists
        if not os.path.exists(self.json_dataset_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_dataset_path}")

        # Create a dataset of single-turn examples
        self.turn_dataset = None
        self.create_turn_dataset()

        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Load and split the dataset into train, validation, and test sets
        self.load_and_split_datasets()

        print(f"[INFO] TurnDataLoader initialized with {len(self.turn_dataset)} windows.")
        print(f"[INFO] Sizes of datasets:\n"
              f"\ttrain dataset: {len(self.train_dataset) if self.train_dataset else 0},\n"
              f"\tval dataset: {len(self.val_dataset) if self.val_dataset else 0},\n"
              f"\ttest dataset: {len(self.test_dataset) if self.test_dataset else 0}\n\n")


    def create_turn_dataset(self):
        """
        Creates a dataset of single-turn examples from the conversation dataset.
        """

        # Load the dataset from the JSON file
        with open(self.json_dataset_path, 'r', encoding='utf-8') as f:
            conv_dataset = json.load(f)

        dataset = []
        loop = tqdm(conv_dataset, desc="[INFO] Extracting conversation turns", unit="sample")
        for sample in loop:
            for turn in sample['conversation']:
                text = turn['text']
                label = sample['person_couple']
                dataset.append((text, label))

        self.turn_dataset = dataset


    def load_and_split_datasets(self):
        """
        Loads the dataset and splits it into train, validation, and test sets.
        """

        texts = [turn[0] for turn in self.turn_dataset]
        raw_labels = [turn[1] for turn in self.turn_dataset]
        labels = self.label_encoder.fit_transform(raw_labels)

        # Split the dataset into train, validation, and test sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state, stratify=train_labels
        )

        self.train_dataset = ConversationDataset(train_texts, train_labels, self.tokenizer, self.max_length, self.add_token_type_ids)
        self.val_dataset = ConversationDataset(val_texts, val_labels, self.tokenizer, self.max_length, self.add_token_type_ids)
        self.test_dataset = ConversationDataset(test_texts, test_labels, self.tokenizer, self.max_length, self.add_token_type_ids)


    def get_datasets(self):
        """
        Returns the train, validation, and test datasets.
        """
        return self.train_dataset, self.val_dataset, self.test_dataset
    

    def get_label_encoder(self):
        """
        Returns the label encoder used for encoding labels.
        """
        return self.label_encoder
    

    def get_num_labels(self):
        """
        Returns the number of unique labels in the dataset.
        """
        return len(self.label_encoder.classes_)
    

    def get_label_names(self):
        """
        Returns the names of the labels in the dataset.
        """
        return self.label_encoder.classes_.tolist()