
# Char-Level, TF Dev Cert/NLP/generate_text_char_level.py
# Word-Level, TF Dev Cert/NLP/generate_text_word_level.py

#################################################### Example: Char-Level

https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_5.pdf

https://juejin.cn/post/6995777526308012069

https://clownote.github.io/2020/08/20/DeepLearningWithPython/Deep-Learning%20with-Python-ch8_1/

https://blog.csdn.net/weixin_46489969/article/details/125525879


##### run "TF Dev Cert/NLP/generate_text_char_level.py"



Build 371778 samples by using the 1115394 chars in the text
                          feature               label
1 Sample:  60 characters -> 1 character

Input: one-hot encoding (no embedding layer), 39 values [ 0 0 0 0 … 1 .. 0 0 0 ]

input_shape=(60, 39) # 39 input values for each call and call 60 times - 1 sample

LSTM Layer: 128 status values ( 128 neurons )
Parameters:  (39 W + 128 W + 1b) x 128 x 4 = 86016  # 4 times of the RNN's parameters

SoftMax Layer: 39 probability values
Parameters: (128 W + 1 b) x 39 = 5031


Prediction: probability -> next char,  one-shot by using  Multinomial Distribution. 


1 Epoch without GPU:


1 Epoch with GPU:
















# Bernouli Distribution





# Binomial Distribution - flip a coin
https://www.investopedia.com/terms/b/binomialdistribution.asp













# Multinomial Distribution - roll a dice
https://www.statisticshowto.com/multinomial-distribution/
https://en.wikipedia.org/wiki/Multinomial_distribution







# numpy.random.multinomial
https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html





# numpy.argmax
https://numpy.org/doc/stable/reference/generated/numpy.argmax.html





#### Old

5000 unique words；
1 sentence： 500 words，32 values per word

Embedding layer: 5000 x 32 = 160000 parameters

LR Layer: 500x32 + 1 = 16,001 parameters
 
Embedding layer: 5000 x 32 = 160000 parameters
RNN Layer: 32 x ( 32 + 32 + 1 ) = 2,080 parameters, 

Embedding layer: 5000 x 32 = 160000 parameters
LR Layer: 4 x 32 x ( 32 + 32 + 1 ) = 8,320 parameters



# CHN
https://zhuanlan.zhihu.com/p/31656392
https://blog.csdn.net/newlw/article/details/122546868




#################################################### Example: Word-Level
https://www.kaggle.com/datasets/aashita/nyt-comments
https://www.kaggle.com/datasets/aashita/nyt-comments?resource=download
https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms



##### run "TF Dev Cert/NLP/generate_text_word_level.py"





831 headlines with 2421 unique words ( ID: 1 ~ 2421)；
ID: 0 is reserved, so totally 2422 words;

Generate 4806 samples:
Input: Sequence of ID, 18 IDs ( including ID:0  )
Output: One-hot, 2422 classes )




https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer




# keras.preprocessing.text.Tokenizer
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer




# fit_on_texts
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#fit_on_texts





# ngram
https://www.mathworks.com/discovery/ngram.html




# pad_sequences
https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences





# to_categorical
https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical





# Embedding Layer
https://www.tensorflow.org/text/guide/word_embeddings#:~:text=An%20embedding%20is%20a%20dense,weights%20for%20a%20dense%20layer
https://keras.io/api/layers/core_layers/embedding/

https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce







#################################################### Summary
#################################################### Summary


	 Char-Level	Word-Level
Input	371778 samples:	4806 samples:
	60 chars ------------------------> 1 char	18 words ------------------------> 1 word
	One-hot,39                           One-hot, 39	Sequence of ID                    One-hot,2422
Model	model = Sequential()	model = Sequential()
		
		model.add(  Embedding(2422, 10, input_length=18)  )
		# totally 2422 words and every word vector has 10 components;
		# 2422 x 10 = 24,220 parameters
		
		
	model.add(  LSTM( 128, input_shape=(60, 39) )  ) 	
	# the State H contains 128 values	model.add(  LSTM(100)  )   # input_shape=(18,10)
	# Inputs:  39 values for a char (one-hot), 60 times	# the State H has 100 values
	# RNN: (39 W + 128 W + 1b) x 128 x 1 = 21504 Parameters	# Inputs: 10 values for a word, 18 times
	# LSTM: (39 W + 128 W + 1b) x 128 x 4 = 86016 Parameters	# RNN: (10 W + 100 W + 1b) x 100 x 1 = 11100 parameters
		# LSTM: (10 W + 100 W + 1b) x 100 x 4 = 44400 parameters
		
		model.add(  Dropout(0.1)  )
		
	model.add(  Dense(39, activation='softmax')  )	
	# Softmax	model.add(  Dense(2422, activation='softmax')  )
	# Inputs: 128 values from the last State H	# Softmax
	# Outputs: 39 P values	# Inputs: 100 values from the last State H
	# Parameters: (128 W + 1 b) x 39 = 5031	# Outputs: 2422 values
		# Parameters: (100 W + 1 b) x 2422 = 244622
		
		
		
		
		
		



#################################################### NLTK - Natural Language Toolkit

https://www.nltk.org/





#################################################### Example: Word-Level 
https://towardsdatascience.com/word-and-character-based-lstms-12eb65f779c2
https://github.com/ruthussanketh/natural-language-processing/blob/main/word-and-character-LSTM/corpus.txt

https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb


https://towardsdatascience.com/text-generation-gpt-2-lstm-markov-chain-9ea371820e1e



https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms

https://www.mathworks.com/help/deeplearning/ug/word-by-word-text-generation-using-deep-learning.html



#################################################### Example: Char-Level 
https://www.tensorflow.org/text/tutorials/text_generation
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://towardsdatascience.com/text-generation-using-rnns-fdb03a010b9f
https://www.analyticsvidhya.com/blog/2022/02/explaining-text-generation-with-lstm/








# ln
https://www.medcalc.org/manual/ln-function.php





# Logit in Math

https://en.wikipedia.org/wiki/Logit
https://deepai.org/machine-learning-glossary-and-terms/logit
https://lucasdavid.github.io/blog/machine-learning/crossentropy-and-logits/










Sigmod:






# Logit in Machine Learning
https://developers.google.com/machine-learning/glossary/#logits
https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow

https://www.cnblogs.com/SupremeBoy/p/12266155.html











# from_logits
https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function




#tf.function










# RNN many-to-many loss 
https://www.cnblogs.com/tangweijqxx/p/10637396.html
https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/09/01-RNN-Many-to-many.html





# Ragged Tensor
https://www.tensorflow.org/guide/ragged_tensor




# tf.strings.unicode_split
https://www.tensorflow.org/api_docs/python/tf/strings/unicode_split




# tf.keras.layers.StringLookup  
https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup?version=nightly

 /TF Advanced/ProprocessingWithStatus.py


# tf.data.Dataset.from_tensor_slices
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#from_tensor_slices


# dataset map
https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#map






b'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '

b'are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\nFirst Citizen:\nFirst, you k'

b"now Caius Marcius is chief enemy to the people.\n\nAll:\nWe know't, we know't.\n\nFirst Citizen:\nLet us ki"

b"ll him, and we'll have corn at our own price.\nIs't a verdict?\n\nAll:\nNo more talking on't; let it be d"

b'one: away, away!\n\nSecond Citizen:\nOne word, good citizens.\n\nFirst Citizen:\nWe are accounted poor citi'
![Uploading image.png…]()
