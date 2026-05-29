# router.py
import re


class LatticeRouter:
    def __init__(self, orchestrator):
        # The router links directly to the orchestrator wrapper instance
        self.orchestrator = orchestrator
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
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        return set(clean_text.split()) - self.stop_words

    def plan_path(self, user_intent: str) -> list:
        user_tokens = self.tokenize(user_intent)
        matched_cells = []

        # DYNAMIC ACCESSIBILITY: Query the wrapper to get live available cell options
        available_cells = self.orchestrator.get_all_available_cells()

        for cell in available_cells:
            intersection = user_tokens.intersection(cell.keywords)
            union = user_tokens.union(cell.keywords)
            similarity_score = len(intersection) / len(union) if len(union) > 0 else 0.0

            if similarity_score > 0.0:
                matched_cells.append((cell, similarity_score))

        matched_cells.sort(key=lambda item: item[0].stage)
        return [item[0] for item in matched_cells]
