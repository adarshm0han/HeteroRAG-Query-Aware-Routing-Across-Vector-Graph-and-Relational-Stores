from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pipeline import QUERY_PREFIX
from retriever import VectorRetriever


def test_search_embeds_query_filters_tables_and_preserves_evidence():
    retriever = VectorRetriever.__new__(VectorRetriever)
    retriever.model = Mock()
    retriever.model.encode.return_value = np.array([[0.1, 0.2]])
    metadata = {'text': 'A passage', 'passage_id': '/wiki/X', 'title': 'X',
                'table_ids': ['Table_0'], 'table_count': 1.0}
    retriever.index = Mock()
    retriever.index.query.return_value = SimpleNamespace(matches=[SimpleNamespace(metadata=metadata, score=0.8)])
    result = retriever.search('A question?', k=5, table_ids=['Table_0'])
    retriever.model.encode.assert_called_once_with([QUERY_PREFIX + 'A question?'], normalize_embeddings=True)
    retriever.index.query.assert_called_once_with(vector=[0.1, 0.2], top_k=5, include_metadata=True,
                                                filter={'table_ids': {'$in': ['Table_0']}})
    assert result == [{**metadata, 'table_count': 1, 'score': 0.8, 'source': 'vector'}]
    retriever.search('A question?')
    assert retriever.index.query.call_args.kwargs['filter'] is None
