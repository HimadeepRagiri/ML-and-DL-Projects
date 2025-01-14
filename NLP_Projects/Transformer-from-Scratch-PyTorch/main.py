import torch
from transformer import Transformer
from utils import create_pad_mask, create_subsequent_mask

if __name__ == "__main__":
    # Parameters
    src_vocab_size = 50
    tgt_vocab_size = 50
    max_len = 10
    d_model = 16
    num_heads = 4
    d_ff = 64
    num_layers = 2
    batch_size = 2

    # Initialize model
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len
    )

    # Create synthetic data
    src = torch.randint(0, src_vocab_size, (batch_size, max_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, max_len))

    # Create masks
    src_mask = create_pad_mask(src)
    tgt_mask = create_pad_mask(tgt) & create_subsequent_mask(max_len)

    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)

    # Print results
    print("Output shape:", output.shape)

