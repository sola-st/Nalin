"""

Created on 25-June-2020
@author Jibesh Patra


"""

import fasttext

# Skipgram model :
print("Learning embeddings")
model = fasttext.train_unsupervised('benchmark/all_tokens', model='skipgram')

model.save_model("results/python_embeddings.bin")
