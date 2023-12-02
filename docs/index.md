# Welcome to airoboros

NOTE: I'm trying to make this project sound as exciting as it is while also been informative for a technical audience 

TAGLINE: **Multiple experts are better than a generalist**

OR

TAGLINE: **Making ChatGPT fit on your laptop**

---

Airoboros creates multiple lightweight Large Language Models (LLMs), each an expert 
in a specific area. These models are then queryable via the OpenAI-compatible API. 
The API will dynamically load the best model for the query received.

Find out more below, or get going with the [installation] and [quick-start].

* [What are lightweight models](explanation/lightweight-models.md)?
* [How are these models created](explanation/model-creation.md)?
* [How are these models queried](explanation/model-querying.md)?


Expert models are created by optimising existing models, such as llama-2. 
These smaller models can perform as well as much larger models, yet have 
far lower hardware requirements.
This approach was inspired by the LoRA paper {!docs/includes/papers/lora.md!}.

