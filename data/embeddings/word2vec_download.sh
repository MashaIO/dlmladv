# Script to download and extract Word2Vec
# word embeddings. More information at:
# https://code.google.com/archive/p/word2vec/

URL=https://s3.amazonaws.com/dl4j-distribution
EMBEDDINGS=GoogleNews-vectors-negative300.bin

# download and extract
if [ ! -f $EMBEDDINGS ]; then
  wget -c "$URL/$EMBEDDINGS.gz"
  gunzip $EMBEDDINGS.gz
fi
