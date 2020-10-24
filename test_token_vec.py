import gensim
if __name__ == "__main__":
    path = "./data/token_vec_300.bin"
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    print(model.get_vector("ne"))
 #   with open(path, "rb") as f:

