from services.cost_tracker import calculate_chat_cost, calculate_embedding_cost, MODEL_PRICING


def test_calculate_chat_cost_known_model():
    cost = calculate_chat_cost("qwen/qwen3.6-27b", input_tokens=1_000_000, output_tokens=0)
    assert cost == 0.30


def test_calculate_chat_cost_output_tokens():
    cost = calculate_chat_cost("qwen/qwen3.6-27b", input_tokens=0, output_tokens=1_000_000)
    assert cost == 2.00


def test_calculate_chat_cost_unknown_model_returns_zero():
    cost = calculate_chat_cost("unknown/model-xyz", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0


def test_calculate_embedding_cost():
    cost = calculate_embedding_cost("openai/text-embedding-3-small", tokens=1_000_000)
    assert cost == 0.02


def test_pricing_table_has_both_models_in_use():
    assert "qwen/qwen3.6-27b" in MODEL_PRICING
    assert "openai/text-embedding-3-small" in MODEL_PRICING
