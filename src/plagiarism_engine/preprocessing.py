import re
import string

# A basic set of English stop words based on project requirements
STOP_WORDS = {"the", "is", "and", "a", "an", "of", "to", "in", "for", "with", "on", "it", "this", "that"}

def clean_and_tokenize(text: str) -> list:
    """
    Cleans the input text by normalizing case, removing punctuation, 
    handling extra spaces, and filtering out common stop words.
    Handles empty, very short, or invalid input texts safely.
    """
    # 1. Management of empty, short, or invalid texts
    if not text or not isinstance(text, str) or len(text.strip()) < 2:
        return []

    # 2. Case normalization (convert to lowercase)
    text = text.lower()

    # 3. Remove punctuation by replacing them with spaces
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    # 4. Tokenization and removing extra spaces
    tokens = text.split()

    # 5. Remove highly frequent stop words
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]

    return filtered_tokens

def generate_word_shingles(tokens: list, shingle_size: int = 3) -> set:
    """
    Converts a list of tokens into word shingles (n-grams) of a specific size.
    Returns a set of unique shingles.
    """
    # Handle cases where token count is less than the requested shingle size
    if not tokens or len(tokens) < shingle_size:
        return set()

    shingles = set()
    for i in range(len(tokens) - shingle_size + 1):
        # Create a shingle by joining consecutive words with a single space
        shingle = " ".join(tokens[i:i + shingle_size])
        shingles.add(shingle)
        
    return shingles