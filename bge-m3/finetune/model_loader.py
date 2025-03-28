from transformers import AutoTokenizer, AutoModel

def load_model(model_name, device="cuda"):
    """
    Args:
        model_name (str): Hugging Face ID
    Returns:
        tuple: tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model
