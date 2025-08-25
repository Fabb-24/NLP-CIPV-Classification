import re
import spacy

nlp = spacy.load("it_core_news_sm")


TURN_SEPARATOR = r"\s{5,}"


def clean_turn(turn: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    """
    Cleans a conversation turn: remove names, index, punctuation, lowercases, and optionally removes stopwords.
    
    Args:
        turn (str): The conversation turn to clean.
    
    Returns:
        str: The cleaned turn.
    """
    
    turn = turn.strip()

    # Remove numeration ("1)", "2.") or names ("Mario:", "Giulia:") at the beginning
    turn = re.sub(r'^\s*(\d+[\)\.]|[A-Z][a-zA-Z]{1,15}:)', '', turn)
    # Remove double quotes and other quotation marks
    turn = turn.replace('"', '').replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    # Lowercases
    turn = turn.lower()
    # Remove extra spaces
    turn = re.sub(r'\s+', ' ', turn).strip()
    # Tokenize the turn
    tokens = turn.split()
    # Remove too short turns
    if len(tokens) < 2:
        return ""
    turn = ' '.join(tokens)

    if remove_stopwords or lemmatize:
        doc = nlp(turn)
        tokens = []
        for token in doc:
            if remove_stopwords and token.is_stop:
                continue
            token_text = token.lemma_ if lemmatize else token.text
            tokens.append(token_text)
        turn = ' '.join(tokens)

    return turn


def preprocess_conversation(text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    """
    Preprocesses a conversation: divides into turns, cleans each turn, and joins them back.

    Args:
        text (str): The conversation text.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to apply lemmatization.

    Returns:
        str: The preprocessed conversation text.
    """

    turns = re.split(TURN_SEPARATOR, text)
    cleaned_turns = [clean_turn(turn, remove_stopwords=remove_stopwords, lemmatize=lemmatize) for turn in turns if turn.strip()]
    cleaned_turns = [turn for turn in cleaned_turns if turn]  # Remove empty turns
    return ' [TURN_SEP] '.join(cleaned_turns)


def conversation_to_windows(conversation: dict, window_size: int = 3, remove_stopwords: bool = False, lemmatize: bool = False) -> list:
    """
    Converts a conversation dictionary into a list of windows of conversation turns.

    Args:
        conversation (dict): The conversation dictionary containing turns.
        window_size (int): The number of turns to include in each window.
        remove_stopwords (bool): Whether to remove stopwords from the text.
        lemmatize (bool): Whether to apply lemmatization to the tokens.

    Returns:
        list: A list of tuples, each containing a window string and the person couple information.
    """

    windows = []
    turns = conversation["conversation"]

    # Create windows of conversation turns
    for i, turn in enumerate(turns):
        # create a window containing the elements from i to i + window_size if possible
        if i + window_size <= len(turns):
            window = ""

            # Construct the window string with speaker and turn separator tags
            for j in range(i, i + window_size):
                window += "[SPEAKER1] " if turns[j]["speaker"] == conversation['name1'] else "[SPEAKER2] "
                text = turns[j]["text"]

                # Apply lemmatization and stopword removal if specified
                doc_turn = nlp(text)
                tokens = []
                for token in doc_turn:
                    if token.is_punct or token.is_space:
                        continue
                    if remove_stopwords and token.is_stop:
                        continue
                    token_text = token.lemma_ if lemmatize else token.text
                    tokens.append(token_text)
                text = ' '.join(tokens)

                window += text

                if j < i + window_size - 1:
                    window += " [TURN_SEP] "

            # Append the window with the person couple information
            windows.append((window, conversation["person_couple"]))

    return windows