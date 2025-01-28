import torch


def generate_sequence(model, start_sequence, max_length, device):
    """
    Generate a sequence continuation using the trained GPT model.

    Args:
        model: Trained GPT model
        start_sequence: Initial sequence to continue (torch.Tensor)
        max_length: Maximum length of the generated sequence
        device: Device to run the model on

    Returns:
        Complete generated sequence including the start sequence
    """
    model.eval()
    with torch.no_grad():
        # Initialize sequence with start_sequence
        current_sequence = start_sequence.clone()

        # Generate new tokens one at a time
        for _ in range(max_length - len(start_sequence)):
            # Get model prediction
            output = model(current_sequence.unsqueeze(0).to(device))

            # Get the last token prediction
            next_token_logits = output[0, -1, :]

            # Get the most likely next token
            next_token = torch.argmax(next_token_logits).unsqueeze(0)

            # Append the new token to the sequence
            current_sequence = torch.cat([current_sequence, next_token.cpu()])

    return current_sequence


def make_predictions(model, hp):
    """
    Make and display several example predictions.

    Args:
        model: Trained GPT model
        hp: Hyperparameters object
    """
    model.eval()

    print("\nMaking predictions with trained model:")
    print("-" * 50)

    # Test cases with different starting sequences
    test_cases = [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([10, 11, 12], dtype=torch.long),
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
        torch.tensor([20, 21], dtype=torch.long)
    ]

    for i, start_seq in enumerate(test_cases, 1):
        # Generate continuation
        generated = generate_sequence(model, start_seq, hp.max_length, hp.device)

        print(f"\nTest Case {i}:")
        print(f"Start sequence:     {start_seq.tolist()}")
        print(f"Full generation:    {generated.tolist()}")
        print(f"Generated portion:  {generated[len(start_seq):].tolist()}")

        # Check if the pattern follows the expected sequence
        expected = torch.tensor([(x + 1) % hp.vocab_size for x in start_seq[:-1]])
        actual = start_seq[1:]
        accuracy = (expected == actual).float().mean()
        print(f"Pattern accuracy:   {accuracy:.2%}")