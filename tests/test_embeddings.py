"""Test logic batching của embed_texts, KHÔNG gọi API thật (dùng mock)."""
from unittest.mock import patch
from rag.embeddings import embed_texts


def test_empty_input_returns_empty_list():
    assert embed_texts([]) == []


def test_batching_splits_correctly():
    """Với 250 texts và batch_size=100, phải gọi _embed_batch đúng 3 lần."""
    fake_texts = [f"text {i}" for i in range(250)]

    with patch("rag.embeddings._embed_batch") as mock_embed:
        # Mock trả về vector giả cho mỗi batch call
        mock_embed.side_effect = lambda batch: [[0.1, 0.2, 0.3] for _ in batch]

        result = embed_texts(fake_texts, batch_size=100)

        assert mock_embed.call_count == 3  # 100 + 100 + 50
        assert len(result) == 250


def test_order_preserved_across_batches():
    fake_texts = [f"text_{i}" for i in range(5)]

    with patch("rag.embeddings._embed_batch") as mock_embed:
        # Trả về embedding có giá trị = index để verify thứ tự
        def fake_embed(batch):
            return [[float(fake_texts.index(t))] for t in batch]
        mock_embed.side_effect = fake_embed

        result = embed_texts(fake_texts, batch_size=2)

        assert result[0][0] == 0.0
        assert result[4][0] == 4.0
