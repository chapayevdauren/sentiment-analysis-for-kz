from gensim.models import KeyedVectors

# Creating the model
# Takes a lot of time depending on the vector file size
kk_model = KeyedVectors.load_word2vec_format('data/cc.kk.300.vec')

# Getting the tokens
words = []
for word in kk_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out the dimension of a word vector
print("Dimension of a word vector: {}".format(
    len(kk_model[words[0]])
))

# Print out the vector of a word
# print("Vector components of a word: {}".format(
#     kk_model[words[0]]
# ))

# Pick a word
find_similar_to = 'ит'

# Finding out similar words [default= top 10]
for similar_word in kk_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))

# Test words
word_add = ['қазақстан', 'астана']
word_sub = ['алматы']

# Word vector addition and subtraction
for resultant_word in kk_model.most_similar(
    positive=word_add, negative=word_sub
):
    print("Word : {0} , Similarity: {1:.2f}".format(
        resultant_word[0], resultant_word[1]
    ))
