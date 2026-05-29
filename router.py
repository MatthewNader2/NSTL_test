# router.py
import re

from lattice import GLOBAL_LATTICE


class LatticeRouter:
    def __init__(self):
        self.available_cells = []
        for neighborhood in GLOBAL_LATTICE.values():
            for cell in neighborhood.micro_cells.values():
                self.available_cells.append(cell)

        # English stop words to filter out noise from prompts
        self.stop_words = {
            "i",
            "need",
            "to",
            "a",
            "the",
            "and",
            "my",
            "this",
            "out",
            "any",
            "it",
            "for",
            "with",
            "an",
        }

    def tokenize(self, text: str) -> set:
        """A real text normalizer and tokenizer processing step."""
        # Remove punctuation, cast to lowercase, and split by whitespace
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = set(clean_text.split())
        # Filter out grammatical stop words to extract pure semantic intent
        return tokens - self.stop_words

    def plan_path(self, user_intent: str) -> list:
        user_tokens = self.tokenize(user_intent)
        print(f"[TOKENIZER OUTPUT] Active Prompt Tokens: {list(user_tokens)}")

        matched_cells = []

        # Calculate Jaccard similarity threshold overlap for every cell in system
        for cell in self.available_cells:
            intersection = user_tokens.intersection(cell.keywords)
            union = user_tokens.union(cell.keywords)

            similarity_score = len(intersection) / len(union) if len(union) > 0 else 0.0

            # If there is any structural keyword overlap, mark this cell as a candidate
            if similarity_score > 0.0:
                matched_cells.append((cell, similarity_score))

        # Sort matched cells primarily by their architectural Pipeline Stage
        # This acts as our topological pipeline assembler
        matched_cells.sort(key=lambda item: item[0].stage)

        # Extract the sorted cell references from score wrappers
        final_path = [item[0] for item in matched_cells]

        print(
            f"[ROUTER PLAN GENERATED] Dynamic Path: {[cell.cell_id for cell in final_path]}"
        )
        return final_path
