import json
import os
import pandas as pd
import re
import spacy
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('italian'))
nlp = spacy.load("it_core_news_sm")


TURN_SEPARATOR = r"\s{5,}"


def get_root_path() -> str:
    """
    Returns the root path of the project.
    """
    
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def clean_turn(turn: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    """
    Cleans a conversation turn: remove names, index, punctuation, lowercases, and optionally removes stopwords.
    
    Args:
        turn (str): The conversation turn to clean.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to lemmatize the tokens.
    
    Returns:
        str: The cleaned turn.
    """
    
    turn = turn.strip()
    
    turn = re.sub(r'^\s*(\d+[\)\.]|[A-Z][a-zA-Z]{1,15}:)', '', turn)  # Remove numeration ("1)", "2.") or names ("Mario:", "Giulia:") at the beginning
    turn = turn.replace('"', '').replace("“", "").replace("”", "").replace("‘", "").replace("’", "").replace("'", " ")  # Remove double quotes and other quotation marks
    turn = turn.lower()  # Lowercases
    turn = re.sub(r'[^\w\s]', '', turn)  # Remove punctuation
    turn = re.sub(r'\s+', ' ', turn).strip()  # Remove extra spaces
    
    tokens = turn.split()  # Tokenize the turn
    if len(tokens) < 2:  # Remove too short turns
        return ""
    
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    
    text = ' '.join(tokens)  # Join tokens back into a single string

    if lemmatize:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_space and not token.is_punct]
        text = ' '.join(tokens)

    return text


def preprocess_conversation(text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    """
    Preprocesses a conversation: divides into turns, cleans each turn, and joins them back.

    Args:
        text (str): The conversation text.
        remove_stopwords (bool): Whether to remove stopwords.

    Returns:
        str: The preprocessed conversation text.
    """

    turns = re.split(TURN_SEPARATOR, text)
    cleaned_turns = [clean_turn(turn, remove_stopwords=remove_stopwords, lemmatize=lemmatize) for turn in turns if turn.strip()]
    cleaned_turns = [turn for turn in cleaned_turns if turn]  # Remove empty turns
    return ' [SEP] '.join(cleaned_turns)


def get_conversation_tokens(conversation: str, remove_stopwords: bool = False, lemmatize: bool = False) -> list:
    """
    Splits a conversation into tokens after preprocessing.

    Args:
        conversation (str): The conversation text.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to lemmatize the tokens.

    Returns:
        list: A list of tokens from the preprocessed conversation.
    """
    
    preprocessed_conversation = preprocess_conversation(conversation, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    turns = preprocessed_conversation.split(' [SEP] ')
    tokens = []
    for turn in turns:
        tokens.extend(turn.split())
    return tokens


def convert_dataset_to_json(csv_path: str, json_path: str):
    """
    Converts a CSV file to a JSON file with conversation turns and saves it to the specified output path.

    Args:
        csv_path (str): The path to the input CSV file.
        json_path (str): The path to the output JSON file.
    """

    print(f"[INFO] Converting dataset from {csv_path} to {json_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

    # Read the CSV file and initialize an empty list for JSON data
    df = pd.read_csv(csv_path)
    json_data = []

    # Iterate through each row in the DataFrame
    loop = tqdm(df.iterrows(), total=len(df), desc="[INFO] Converting dataset")
    for _, row in loop:
        # Create a sample dictionary for each row and insert all fields except the conversation
        sample = {}
        name1 = row["name1"]
        name2 = row["name2"]

        sample["person_couple"] = row["person_couple"]
        sample["name1"] = name1
        sample["name2"] = name2
        sample["explaination"] = row["explaination"]
        sample["toxic"] = row["toxic"]
        
        # Split the conversation into turns and clean them
        conversation = row["conversation"]
        turns = re.split(TURN_SEPARATOR, conversation)
        cleaned_turns = [clean_turn(turn) for turn in turns if clean_turn(turn)]

        # Create a conversation structure with alternating names
        curr_name = name1
        conversation_turns = []
        for turn in cleaned_turns:
            conversation_turns.append({
                "speaker": curr_name,
                "text": turn
            })
            curr_name = name2 if curr_name == name1 else name1

        # Add the conversation turns to the sample and append to JSON data
        sample["conversation"] = conversation_turns
        json_data.append(sample)

    # Save the JSON data to a file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Dataset converted and saved to {json_path}.\n\n")