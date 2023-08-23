import json
import faiss
import os
import math
import numpy as np
import glob
import random
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Any, List
from airoboros.embeddings import calculate_embeddings


MAX_LENGTH = 512


class Router:
    """Simple embedding similarity based router.  Allows immediate swap-in
    of LoRAs, so long as the data is available.
    """

    def __init__(
        self,
        model_name_or_path: str = "thenlper/gte-small",
        input_paths: List[str] = [],
        k: int = 50,
        max_samples: int = 500,
    ):
        """Constructor."""
        self.model = SentenceTransformer(model_name_or_path, device="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.k = k
        self.max_samples = max_samples
        if not input_paths:
            input_paths = [
                str(path)
                for path in glob.glob(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "routing_data",
                        "*.jsonl",
                    )
                )
                if str(path).split("/")[-1].startswith("expert_")
            ]
        self.indices = {}
        for path in input_paths:
            expert = path.split("expert_")[-1].split(".jsonl")[0]
            self.indices[expert] = self.create_index(path)

    def create_index(self, input_path: str) -> Any:
        """Create a faiss index from the routing data for a given expert."""
        logger.info(f"Creating routing faiss index: {input_path}")
        index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        all_items = []
        with open(input_path, "r") as infile:
            for line in infile.readlines():
                all_items.append(json.loads(line)["instruction"])
        random.shuffle(all_items)
        for item in tqdm(all_items[0 : self.max_samples]):
            index.add(
                np.array([calculate_embeddings(item, self.model, self.tokenizer)])
            )
        return index

    def route(self, prompt: str) -> str:
        """Select the model to route incoming requests to, based on faiss index avg distances."""
        query_emb = np.array([calculate_embeddings(prompt, self.model, self.tokenizer)])
        best_expert = None
        best_distance = math.inf
        for expert, index in self.indices.items():
            distances, _ = index.search(query_emb, k=min(index.ntotal, self.k))
            distances = distances[0].tolist()
            average_distance = sum(distances) / len(distances)
            logger.debug(f"Average distance [{expert}]: {average_distance}")
            if average_distance < best_distance:
                best_distance = average_distance
                best_expert = expert
        logger.success(f"Routing to {best_expert} with score: {best_distance}")
        return best_expert
