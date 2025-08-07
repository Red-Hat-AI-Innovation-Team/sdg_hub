import random

# Expanded list of descriptive adjectives
ADJECTIVES = [
    "amber", "azure", "blue", "bronze", "crimson", "cyan", "golden", "green",
    "indigo", "jade", "purple", "ruby", "silver", "teal", "violet",    
    "bold", "brave", "bright", "calm", "clever", "cosmic", "deep", "eager",
    "fair", "fierce", "gentle", "happy", "keen", "kind", "light", "noble",
    "proud", "quick", "quiet", "rapid", "sharp", "silent", "smart", "swift",
    "warm", "wise", "wild", "young",
    "autumn", "desert", "forest", "lunar", "misty", "ocean", "polar", "solar",
    "spring", "summer", "winter",
    "crystal", "flame", "frost", "metal", "storm", "thunder",
    "agile", "daring", "elegant", "graceful", "mighty", "nimble", "precise",
    "steady", "vibrant", "vigilant"
]

# Expanded list of nouns
NOUNS = [
    "bear", "bird", "deer", "dove", "duck", "eagle", "falcon", "fish", "fox",
    "hawk", "heron", "lion", "lynx", "owl", "panda", "raven", "seal", "swan",
    "tiger", "wolf",
    "brook", "cloud", "coast", "creek", "dawn", "dune", "field", "fjord",
    "grove", "hill", "lake", "marsh", "moon", "peak", "reef", "river", "rock",
    "shore", "star", "stream", "sun", "vale", "wave", "wood",
    "ash", "birch", "cedar", "elm", "fern", "iris", "lily", "maple", "oak",
    "pine", "rose", "sage", "vine", "yew",
    "breeze", "flame", "frost", "gale", "mist", "rain", "snow", "storm",
    "wind",
    "aeon", "echo", "flux", "nova", "orbit", "pulse", "quark", "void", "zenith"
]


def get_flow_identifier(name: str, max_words=8, max_length=30) -> str:
    """Generate a random adjective-noun pair as a flow identifier.
    
    Args:
        name: Original flow name (kept for compatibility but not used)
        max_words: Not used in this implementation
        max_length: Not used in this implementation
    
    Returns:
        A string in the format "adjective-noun" using random words from predefined lists
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adjective}-{noun}"