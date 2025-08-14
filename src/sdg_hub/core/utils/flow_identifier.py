# Standard
from pathlib import Path
from typing import Dict, List
import hashlib
import random

# Third Party
import yaml

# Cache for loaded word lists to avoid repeated file I/O
_WORD_CACHE: Dict[str, List[str]] = {}


def _load_word_lists() -> Dict[str, List[str]]:
    """Load word lists from YAML configuration file.

    Returns:
        Dictionary containing 'adjectives' and 'nouns' lists

    Raises:
        FileNotFoundError: If the word list file is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    global _WORD_CACHE

    if _WORD_CACHE:
        return _WORD_CACHE

    # Get path to word list file relative to this module
    current_dir = Path(__file__).parent
    words_file = current_dir / "flow_id_words.yaml"

    try:
        with open(words_file, "r", encoding="utf-8") as f:
            word_data = yaml.safe_load(f)

        _WORD_CACHE = {
            "adjectives": word_data["adjectives"],
            "nouns": word_data["nouns"],
        }

        return _WORD_CACHE

    except FileNotFoundError:
        # Fallback to minimal word lists if configuration file is not found
        _WORD_CACHE = {
            "adjectives": ["bright", "calm", "fast", "smart", "quick"],
            "nouns": ["river", "star", "cloud", "moon", "rock"],
        }
        return _WORD_CACHE
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing word list YAML: {e}")
    except KeyError as e:
        raise KeyError(f"Missing required key in word list YAML: {e}")


def get_flow_identifier(name: str) -> str:
    """Generate a deterministic wandb-style flow identifier.

    Creates a human-readable identifier in the format "adjective-noun-number"
    that is deterministic based on the input name. Same name will always
    produce the same identifier.

    Args:
        name: Flow name to generate identifier from

    Returns:
        A string in the format "adjective-noun-number" (e.g., "bright-river-123")

    Examples:
        >>> get_flow_identifier("My Document QA Flow")
        "bright-river-123"
        >>> get_flow_identifier("My Document QA Flow")  # Same input
        "bright-river-123"  # Same output

    Raises:
        FileNotFoundError: If the word list configuration file is not found
        yaml.YAMLError: If the word list YAML file is malformed
    """
    # Load word lists from YAML configuration
    word_lists = _load_word_lists()
    adjectives = word_lists["adjectives"]
    nouns = word_lists["nouns"]

    # Create deterministic seed from name
    seed_value = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_value)

    # Select words and number deterministically
    adjective = rng.choice(adjectives)
    noun = rng.choice(nouns)
    number = rng.randint(1, 999)

    return f"{adjective}-{noun}-{number}"
