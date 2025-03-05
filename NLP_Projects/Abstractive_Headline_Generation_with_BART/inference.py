from config import Config

def generate_headline(article_text, model, tokenizer):
    """
    Generate a headline for a given article text.
    """
    model.eval()
    inputs = tokenizer(article_text,
                       return_tensors="pt",
                       max_length=Config.MAX_INPUT_LENGTH,
                       truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=Config.MAX_TARGET_LENGTH, num_beams=4, early_stopping=True)
    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline
