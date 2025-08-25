import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer
import spacy

from .conversation_dataset import ConversationDataset
from .util import preprocess_conversation

nlp = spacy.load("it_core_news_sm")


class ConversationDataLoader:
    def __init__(self, csv_dataset_path: str, tokenizer: PreTrainedTokenizer, text_column: str = "conversation", label_column: str = "label",
                 test_size: float = 0.2, val_size: float = 0.1, max_length: int = 512, random_state: int = 42, remove_stopwords: bool = True,
                 lemmatize: bool = True, add_token_type_ids: bool = False):
        """
        Initializes the ConversationDataLoader.

        Args:
            csv_dataset_path (str): Path to the CSV file containing conversation data.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing text data.
            text_column (str): Name of the column containing conversation text.
            label_column (str): Name of the column containing labels.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training set to include in the validation split.
            max_length (int): Maximum length of input sequences.
            random_state (int): Random seed for reproducibility.
            remove_stopwords (bool): Whether to remove stopwords during tokenization.
            lemmatize (bool): Whether to lemmatize tokens during tokenization.
            add_token_type_ids (bool): Whether to add token type IDs to the input.
        """

        print(f"[INFO] Initializing ConversationDataLoader with:\n"
              f"\tdataset path: {csv_dataset_path},\n"
              f"\ttest size: {test_size},\n"
              f"\tval size: {val_size},\n"
              f"\tmax length: {max_length},\n"
              f"\trandom state: {random_state},\n"
              f"\tremove stopwords: {remove_stopwords},\n"
              f"\tlemmatize: {lemmatize}\n")

        self.csv_dataset_path = csv_dataset_path
        self.text_column = text_column
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.add_token_type_ids = add_token_type_ids

        # Initialize label encoder
        self.label_encoder = LabelEncoder()

        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Load and split the dataset
        self.load_and_split_datasets()

        print(f"[INFO] ConversationDataLoader initialized with {len(self.train_dataset) if self.train_dataset else 0} training samples.")
        print(f"[INFO] Sizes of datasets:\n"
              f"\ttrain dataset: {len(self.train_dataset) if self.train_dataset else 0},\n"
              f"\tval dataset: {len(self.val_dataset) if self.val_dataset else 0},\n"
              f"\ttest dataset: {len(self.test_dataset) if self.test_dataset else 0}\n\n")


    def load_and_split_datasets(self):
        """
        Loads the dataset from the CSV file, preprocesses the text, and splits it into train, validation, and test sets.
        """

        # Load the dataset
        df = pd.read_csv(self.csv_dataset_path)

        # Check if the specified columns exist in the DataFrame
        if self.text_column not in df.columns or self.label_column not in df.columns:
            raise ValueError(f"Columns '{self.text_column}' or '{self.label_column}' not found in CSV.")
        
        # Extract texts and labels
        texts = df[self.text_column].astype(str).tolist()
        raw_labels = df[self.label_column].astype(str).tolist()
        labels = self.label_encoder.fit_transform(raw_labels)

        # Preprocess the conversation texts
        texts = [preprocess_conversation(text, remove_stopwords=self.remove_stopwords, lemmatize=self.lemmatize) for text in texts]

        # Split the dataset into train, validation, and test sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=self.val_size / (1 - self.test_size), random_state=self.random_state, stratify=train_labels
        )

        self.train_dataset = ConversationDataset(train_texts, train_labels, self.tokenizer, self.max_length, add_token_type_ids=self.add_token_type_ids)
        self.val_dataset = ConversationDataset(val_texts, val_labels, self.tokenizer, self.max_length, add_token_type_ids=self.add_token_type_ids)
        self.test_dataset = ConversationDataset(test_texts, test_labels, self.tokenizer, self.max_length, add_token_type_ids=self.add_token_type_ids)


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
        Returns the list of label names.
        """
        return self.label_encoder.classes_.tolist()