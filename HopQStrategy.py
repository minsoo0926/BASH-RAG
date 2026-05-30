import heapq
import numpy as np
from loguru import logger
from typing import Any, List, Tuple, Dict, Set
from config import expand_logic_query


def _keyword_set(keywords):
    if keywords is None:
        return set()
    if isinstance(keywords, str):
        return {token.lower() for token in keywords.split() if token}
    try:
        return {str(token).lower() for token in keywords if str(token).strip()}
    except TypeError:
        token = str(keywords).strip().lower()
        return {token} if token else set()


def _sparse_similarity(left_keywords, right_keywords):
    left = _keyword_set(left_keywords)
    right = _keyword_set(right_keywords)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dense_similarity(left_emb, right_emb):
    left = np.array(left_emb)
    right = np.array(right_emb)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom < 1e-9:
        return None
    return float(np.dot(left, right) / denom)


def _hybrid_similarity(left_emb, right_emb, left_keywords, right_keywords):
    dense = _dense_similarity(left_emb, right_emb)
    if dense is None:
        return None
    sparse = _sparse_similarity(left_keywords, right_keywords)
    return 0.5 * sparse + 0.5 * dense


class HopQMixin:
    # Stubs for attributes/methods provided by HopRetriever
    topk: int
    max_hop: int
    epsilon: float
    driver: Any

    def process_query(self, query: str) -> Tuple[List, str]:
        raise NotImplementedError

    def search_docs_mock(self, query_embedding: List, query_keywords: str, topk: int) -> Tuple:
        raise NotImplementedError

    def topk_filter(self, sim_dict: Dict[str, float]) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

    def search_docs_hopq(self, query: str) -> Tuple[List[str], List[float]]:
        """Priority-queue graph traversal with explore-exploit scoring (no LLM calls)."""
        query_embedding, query_keywords = self.process_query(query)
        mock_result = self.search_docs_mock(query_embedding, query_keywords, self.topk)
        if mock_result[0] is not None:
            return mock_result
        start_nodes = mock_result[1]  # List[Tuple[Dict, float]]

        # Max-heap via negated scores; counter breaks ties to avoid dict comparison
        H = []
        counter = 0
        C_score: Dict[str, float] = {}

        for node, score in start_nodes[:self.topk]:
            text = node['text']
            heapq.heappush(H, (-score, counter, node))
            counter += 1
            C_score[text] = max(score, C_score.get(text, float('-inf')))

        expanded: Set[str] = set()

        remaining = self.max_hop * self.topk
        with self.driver.session() as session:
            while remaining > 0 and H:
                remaining -= 1

                v = None
                while H:
                    _, _, candidate = heapq.heappop(H)
                    # Skip expanded nodes cuz same v induces same neighbors
                    if candidate['text'] not in expanded:
                        v = candidate
                        break
                if v is None:
                    break

                expanded.add(v['text'])

                # Select next hop neighbors of v using hybrid sparse+dense similarity.
                neighbors = []
                result = session.run(expand_logic_query, {'text': v['text']})
                for record in result:
                    vp = record['logic_node']

                    exploration = _hybrid_similarity(
                        v.get('embed'),
                        vp.get('embed'),
                        v.get('keywords'),
                        vp.get('keywords'),
                    )
                    exploitation = _hybrid_similarity(
                        query_embedding,
                        vp.get('embed'),
                        query_keywords,
                        vp.get('keywords'),
                    )
                    if exploration is None or exploitation is None:
                        continue

                    score = self.epsilon * exploration + (1 - self.epsilon) * exploitation
                    neighbors.append((score, vp))

                if not neighbors:
                    continue

                neighbors.sort(key=lambda x: x[0], reverse=True)

                # Update C_score for all neighbors
                for score, vp in neighbors:
                    C_score[vp['text']] = max(score, C_score.get(vp['text'], float('-inf')))

                # Push best unexpanded neighbor (n++ fallback)
                for score, vp in neighbors:
                    if vp['text'] not in expanded:
                        heapq.heappush(H, (-score, counter, vp))
                        counter += 1
                        break

        logger.info(f"hopq: visited {len(expanded)} nodes, C_score size {len(C_score)}")
        return self.topk_filter(C_score)
