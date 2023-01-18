# Install Spacy
pip install spacy
python -m spacy download en_core_web_sm

# Get FastText and Word2Vec English for future
git clone https://github.com/facebookresearch/fastText.git
pip install fastText/.
# Load model
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip