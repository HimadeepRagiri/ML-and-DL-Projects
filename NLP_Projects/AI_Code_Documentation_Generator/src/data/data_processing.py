import ast
from datasets import load_dataset
import numpy as np

def extract_docstring_and_code(func_string):
    """
    Extract docstring and code from a Python function string.
    Returns a tuple of (docstring, code).
    """
    import ast
    try:
        # Parse the function string
        tree = ast.parse(func_string)
        if not tree.body:
            return "", func_string

        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            return "", func_string

        # Extract docstring if it exists
        docstring = ast.get_docstring(func_def) or ""

        # Remove docstring from the original code
        if docstring:
            lines = func_string.split('\n')
            doc_lines = docstring.count('\n') + 3  # Account for quotes and spacing
            code = '\n'.join(lines[:1] + lines[doc_lines:])  # Keep function definition
        else:
            code = func_string

        return docstring.strip(), code.strip()
    except:
        return "", func_string

def load_and_analyze_data(config):
    """
    Load and analyze the dataset, providing detailed statistics.
    Returns processed training and validation datasets.
    """
    print("Loading CodeSearchNet dataset...")
    dataset = load_dataset(config.dataset_name, "python")

    # Create subsets for training and validation
    train_dataset = dataset["train"].shuffle(seed=42).select(range(config.train_subset_size))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(config.val_subset_size))

    # Process and analyze the data
    print("\nProcessing and analyzing dataset...")
    train_docs = []
    train_codes = []

    for example in train_dataset:
        doc, code = extract_docstring_and_code(example['whole_func_string'])
        if doc:  # Only keep examples with documentation
            train_docs.append(doc)
            train_codes.append(code)

    print("\nDataset Analysis:")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Training examples with documentation: {len(train_docs)}")
    print(f"Documentation rate: {(len(train_docs)/len(train_dataset))*100:.2f}%")

    # Analyze documentation lengths
    doc_lengths = [len(doc.split()) for doc in train_docs]
    code_lengths = [len(code.split()) for code in train_codes]

    print("\nDocumentation Length Statistics (words):")
    print(f"Average length: {np.mean(doc_lengths):.2f}")
    print(f"Median length: {np.median(doc_lengths):.2f}")
    print(f"Max length: {max(doc_lengths)}")
    print(f"Min length: {min(doc_lengths)}")

    # Create new datasets with processed examples
    def create_doc_dataset(examples):
        docs, codes = zip(*[
            extract_docstring_and_code(func_string)
            for func_string in examples['whole_func_string']
        ])
        return {
            'documentation': list(docs),
            'code': list(codes)
        }

    train_dataset = train_dataset.map(
        create_doc_dataset,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        create_doc_dataset,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Filter out examples without documentation
    train_dataset = train_dataset.filter(lambda x: len(x['documentation']) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x['documentation']) > 0)

    return train_dataset, val_dataset

def preprocess_datasets(train_dataset, val_dataset, tokenizer, config):
    """Preprocess the datasets with tokenization."""
    def preprocess_function(examples):
        # Format: <CODE>code<SEP><DOC>documentation
        formatted_texts = [
            f"{config.code_token}{code}{config.sep_token}{config.doc_token}{doc}"
            for code, doc in zip(examples['code'], examples['documentation'])
        ]

        return tokenizer(
            formatted_texts,
            truncation=True,
            max_length=config.max_length,
            padding="max_length"
        )

    print("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    print("Tokenizing validation dataset...")
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    return tokenized_train, tokenized_val
