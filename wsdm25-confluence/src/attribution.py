from copy import deepcopy
from typing import List, Tuple, Union, Any, Dict

import numpy as np
from jsonargparse import CLI
from models.data import Document
from preprocessing.embedding import embed_text
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from utils import chat_with_gpt


def masked_softmax(scores: list[float], temperature=0.1) -> np.ndarray:
    """
    Compute the masked softmax of a list of similarity scores, convert them
    to percentages, and ensure that any original score of 0.0 results in an
    output of 0.0 (if initial contribution is 0.0 then this position in
    the input vector should not play any role in the final output vector)

    Parameters:
    scores (List[float]): A list of similarity scores.

    Returns:
    np.ndarray: A numpy array containing the masked softmax probabilities as
    percentages.
    """
    # Convert scores to a numpy array
    scores = np.array(scores)

    # Create a mask for scores that are zero, actually
    # creates a Boolean array that indicates which elements of
    # the scores array are not equal to 0.0.
    mask = scores != 0.0

    # Apply the mask to the scores - indexing an array with a boolean mask
    # selects elements where the mask is True. The result is a new array that
    # contains only the elements of scores corresponding to
    # True values in the mask. Sample:
    # Scores: [0.  1.5 0.  2.3 4.1]
    # Mask: [False  True False  True  True]
    # Masked Scores: [1.5 2.3 4.1]
    masked_scores = scores[mask]

    # Compute the exponentials of the masked scores
    exp_scores = np.exp(masked_scores / temperature)

    # Compute the sum of the exponentials
    sum_exp_scores = np.sum(exp_scores)

    # Compute the softmax values for the masked scores
    softmax_values = exp_scores / sum_exp_scores

    # Initialize the final percentage values array with zeros
    # np.zeros_like(scores) is used to create an array of zeros that has
    # the same shape as the scores array.
    percentage_values = np.zeros_like(scores)

    # Place the computed softmax percentages in the masked positions. The line
    # percentage_values[mask] = softmax_values * 100 assigns the computed
    # softmax values, converted to percentages, to the appropriate positions
    # in the percentage_values array using the boolean mask.
    percentage_values[mask] = softmax_values * 100

    return percentage_values


class AttributionOutput(BaseModel):
    """
    Represents the output of an attribution calculation.

    Attributes:
    -----------
    attributed_evidences : List[int]
        Indices of evidences (passages) that influence the model's answer.
    answer_counterfactual : str
        The answer generated when the corresponding passages are removed or altered.
    """

    attributed_evidences: List[int]
    answer_counterfactual: str

    def format_doc_probabilities(self, probabilities: np.ndarray) -> str:
        """Format document probabilities with document numbers."""
        formatted = []
        for i, prob in enumerate(probabilities):
            formatted.append(f"Doc {i+1}: {prob:.3f}")
        return "\n".join(formatted)


class PassageCluster:
    """
    Clusters passages based on a chosen distance metric, defaulting to 'cosine'.

    Methods:
    --------
    get_clusters(passages, eps, min_samples):
        Returns a list of cluster labels for each passage. Outliers are labeled '-1'.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    def _get_feature_clusters(
        self, passages: List[str], eps: float = 0.5, min_samples: int = 2
    ) -> np.ndarray:
        """
        Embeds each passage and clusters the embeddings using DBSCAN.

        Parameters:
        -----------
        passages : List[str]
            The passages to cluster.
        eps : float
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.
        min_samples : int
            The number of samples in a neighborhood for a point to be considered
            a core point.

        Returns:
        --------
        np.ndarray
            Array of cluster labels for each passage index.
        """
        embeddings = [embed_text(passage) for passage in passages]

        embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
        embeddings = np.array(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=self.metric)
        labels = clustering.fit_predict(embeddings)
        return labels

    def get_clusters(
        self, passages: List[str], eps: float = 0.5, min_samples: int = 2
    ) -> List[int]:
        """
        Public method to handle clustering and catch any errors.

        Parameters:
        -----------
        passages : List[str]
            The passages to cluster.
        eps : float
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.
        min_samples : int
            The number of samples in a neighborhood for a point to be considered
            a core point.

        Returns:
        --------
        List[int]
            A list of cluster labels for each passage. `-1` denotes an outlier.
        """
        try:
            return self._get_feature_clusters(passages, eps, min_samples).tolist()
        except Exception as e:
            print(f"Error during clustering: \n {e}")
            return [-1 for _ in passages]


class Attribution:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Parameters:
        -----------
        model_name : str
            The model name or identifier for the GPT-like model.
        """
        self.passage_cluster = PassageCluster()
        self.model_name = model_name

    def _evaluate_answer_pair(self, answer: str, counterfactual_answer: str) -> float:
        """
        Evaluates the difference/similarity between the original answer and
        a counterfactual answer by computing embeddings and measuring
        1 - (dot_product).

        Parameters:
        -----------
        answer : str
            The original answer text.
        counterfactual_answer : str
            The answer text generated without certain evidences.

        Returns:
        --------
        float
            A scalar similarity/dissimilarity measure. Larger means more
            dissimilar in this example.
        """
        embedding_answer = embed_text(answer)
        embedding_counterfactual = embed_text(counterfactual_answer)

        # Normalize the embeddings to unit length
        # original work at https://arxiv.org/pdf/2412.10571
        # used JinaAI embeddings
        # (https://huggingface.co/jinaai/jina-embeddings-v3)
        # but here we use OpenAI functions for simplicity
        embedding_answer = embedding_answer / np.linalg.norm(embedding_answer)
        embedding_counterfactual = embedding_counterfactual / np.linalg.norm(
            embedding_counterfactual
        )

        return 1 - float(np.dot(embedding_answer, embedding_counterfactual))

    def _distribute_cluster_probabilities(
        self, cluster_probs: np.ndarray, clusters: List[int]
    ) -> np.ndarray:
        """
        Distributes cluster-level probabilities to individual documents.
        Each document gets the probability of its cluster.

        Parameters:
        -----------
        cluster_probs : np.ndarray
            Probability values for each cluster
        clusters : List[int]
            Cluster assignments for each document

        Returns:
        --------
        np.ndarray
            Document-level probabilities
        """
        # Create a mapping from cluster ID to its probability
        unique_clusters = list(set(clusters))
        cluster_prob_map = {
            cluster_id: cluster_probs[i] for i, cluster_id in enumerate(unique_clusters)
        }

        # Distribute probabilities to documents
        doc_probabilities = np.array([cluster_prob_map[c] for c in clusters])
        return doc_probabilities

    def get_attributions(
        self,
        question: str,
        evidences: List[Document],
        history: List[str],
        answer: Union[str, None] = None,
        eps: float = 0.005,
        min_samples: int = 2,
    ) -> Tuple[List[str], AttributionOutput, Dict[str, List[int]]]:
        """
        Given a question and a set of evidence passages, this method:
        1. Clusters the passages using PassageCluster.
        2. Generates a 'baseline' answer with all evidences.
        3. For each cluster, removes it from the set of evidences and generates
           a counterfactual answer.
        4. Computes which cluster removal leads to the greatest difference
           from the baseline answer (i.e., which cluster is most "attributive").

        Parameters:
        -----------
        question : str
            The user's query.
        evidences : List[Document]
            A list of Document objects containing content for potential answers.
        answer : Union[str, None]
            The answer to the question (if known). If None, the answer will be
            generated using the model.
        history : List[str]
            A list of previous user queries or dialogue context (currently unused
            in the logic but provided for future expansions).

        Returns:
        --------
        AttributionOutput
            The cluster that contributed the most to the final answer
            and the counterfactual answer without that cluster.
        """
        # Cluster the evidences
        evidence_contents = [doc.content for doc in evidences]
        clusters = self.passage_cluster.get_clusters(
            evidence_contents, eps=eps, min_samples=min_samples
        )

        # Convert outliers (-1) to unique new cluster IDs
        unique_cluster_id = max(clusters) + 1 if clusters else 0
        adjusted_clusters = []
        for label in clusters:
            if label == -1:
                adjusted_clusters.append(unique_cluster_id)
                unique_cluster_id += 1
            else:
                adjusted_clusters.append(label)

        # Generate the full answer (with all evidences)
        if answer:
            baseline_answer = answer
        else:
            baseline_completion = chat_with_gpt(
                question, evidences, model=self.model_name
            )
            baseline_answer = baseline_completion.content

        # Generate counterfactual answers per cluster
        cf_results: List[AttributionOutput] = []
        cluster_set = set(adjusted_clusters)
        for cluster_id in cluster_set:
            # Deepcopy the documents to remove the relevant cluster's content
            modified_docs = deepcopy(evidences)
            cluster_indices: List[int] = []

            for idx, label in enumerate(adjusted_clusters):
                if label == cluster_id:
                    # Mark this doc as belonging to the cluster being zeroed
                    cluster_indices.append(idx)
                    modified_docs[idx].content = ""  # Remove/blank out content

            cf_completion = chat_with_gpt(
                question, modified_docs, model=self.model_name
            )
            counterfactual_answer = cf_completion.content

            cf_output = AttributionOutput(
                attributed_evidences=cluster_indices,
                answer_counterfactual=counterfactual_answer,
            )
            cf_results.append(cf_output)

        # Compute which cluster leads to the largest difference from the baseline
        dissimilarities = [
            self._evaluate_answer_pair(baseline_answer, cf.answer_counterfactual)
            for cf in cf_results
        ]

        softmax_output = masked_softmax(dissimilarities)
        max_idx = int(np.argmax(dissimilarities))

        # Distribute cluster probabilities to individual documents
        doc_probabilities = self._distribute_cluster_probabilities(
            softmax_output, adjusted_clusters
        )

        # Create cluster mapping
        cluster_doc_mapping = {}
        for cluster_id in set(adjusted_clusters):
            doc_indices = [
                i + 1 for i, c in enumerate(adjusted_clusters) if c == cluster_id
            ]
            cluster_doc_mapping[f"Cluster {cluster_id}"] = doc_indices

        doc_probabilities_formatted = []
        for i, prob in enumerate(doc_probabilities):
            doc_probabilities_formatted.append(f"Doc {i+1}: {prob:.3f}")

        return doc_probabilities_formatted, cf_results[max_idx], cluster_doc_mapping


def main(model_name: str = "gpt-4o"):
    """
    CLI entry point for demonstration purposes.
    """
    question = "What is the capital of France?"
    history = ["What is the capital of Germany?", "What is the capital of Spain?"]
    documents = [
        "Berlin is the capital of Germany.",
        "I think Berlin is the capital of Germany.",
        "Madrid is the capital of Spain.",
        "Paris is the capital of France.",
    ]

    doc_objects = [Document(content=doc, id="", title="", url="") for doc in documents]

    attributer = Attribution(model_name=model_name)
    softmax_output, attribution_result, cluster_doc_mapping = (
        attributer.get_attributions(question, doc_objects, history)
    )
    print("Evidence Probability:", softmax_output)
    print("Most important evidence:", attribution_result)
    print("Cluster Document Mapping:", cluster_doc_mapping)


if __name__ == "__main__":
    CLI(main, as_positional=False)
