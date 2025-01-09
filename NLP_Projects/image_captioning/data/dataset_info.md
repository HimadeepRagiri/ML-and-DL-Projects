# Dataset Setup Instructions

## Flickr8k Dataset
1. Visit Kaggle dataset page: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Click "Download" (you need to be logged into Kaggle)
3. Once downloaded, extract the contents:
   - Place all images from `Images/` into `data/images/`
   - Place `captions.txt` into `data/`

## GloVe Embeddings
1. Download GloVe embeddings from Stanford NLP:
   - Direct link: https://nlp.stanford.edu/data/glove.6B.zip
2. Extract the downloaded zip file
3. Copy `glove.6B.300d.txt` to `data/`

Note: The complete Flickr8k dataset is approximately 1GB, and the GloVe embeddings file is about 1.2GB. Due to their size, these files are not included in the repository and need to be downloaded separately.
