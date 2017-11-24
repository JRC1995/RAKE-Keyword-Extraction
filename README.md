
# Implementation of RAKE

Based on: 

[Rose, Stuart & Engel, Dave & Cramer, Nick & Cowley, Wendy. (2010). Automatic Keyword Extraction from Individual Documents. Text Mining: Applications and Theory. 1 - 20. 10.1002/9780470689646.ch1.](https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents)

The input text is given below


```python
#Source of text:
#https://www.researchgate.net/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents

Text = "Compatibility of systems of linear constraints over the set of natural numbers. \
Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and \
nonstrict inequations are considered. \
Upper bounds for components of a minimal set of solutions and \
algorithms of construction of minimal generating sets of solutions for all \
types of systems are given. \
These criteria and the corresponding algorithms for constructing \
a minimal supporting set of solutions can be used in solving all the \
considered types of systems and systems of mixed types."
```

The raw input text is cleaned off non-printable characters (if any) and turned into lower case.
The processed input text is then tokenized using NLTK library functions. 


```python

import nltk
from nltk import word_tokenize
import string

#nltk.download('punkt')

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    text = filter(lambda x: x in printable, text) #filter funny characters, if any.
    return text

Cleaned_text = clean(Text)

text = word_tokenize(Cleaned_text)

print "Tokenized Text: \n"
print text
```

    Tokenized Text: 
    
    ['compatibility', 'of', 'systems', 'of', 'linear', 'constraints', 'over', 'the', 'set', 'of', 'natural', 'numbers', '.', 'criteria', 'of', 'compatibility', 'of', 'a', 'system', 'of', 'linear', 'diophantine', 'equations', ',', 'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are', 'considered', '.', 'upper', 'bounds', 'for', 'components', 'of', 'a', 'minimal', 'set', 'of', 'solutions', 'and', 'algorithms', 'of', 'construction', 'of', 'minimal', 'generating', 'sets', 'of', 'solutions', 'for', 'all', 'types', 'of', 'systems', 'are', 'given', '.', 'these', 'criteria', 'and', 'the', 'corresponding', 'algorithms', 'for', 'constructing', 'a', 'minimal', 'supporting', 'set', 'of', 'solutions', 'can', 'be', 'used', 'in', 'solving', 'all', 'the', 'considered', 'types', 'of', 'systems', 'and', 'systems', 'of', 'mixed', 'types', '.']


NLTK is again used for <b>POS tagging</b> the input text.


Description of POS tags: 


http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html


```python
#nltk.download('averaged_perceptron_tagger')
  
POS_tag = nltk.pos_tag(text)

print "Tokenized Text with POS tags: \n"
print POS_tag
```

    Tokenized Text with POS tags: 
    
    [('compatibility', 'NN'), ('of', 'IN'), ('systems', 'NNS'), ('of', 'IN'), ('linear', 'JJ'), ('constraints', 'NNS'), ('over', 'IN'), ('the', 'DT'), ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('numbers', 'NNS'), ('.', '.'), ('criteria', 'NNS'), ('of', 'IN'), ('compatibility', 'NN'), ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('diophantine', 'NN'), ('equations', 'NNS'), (',', ','), ('strict', 'JJ'), ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'), ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.'), ('upper', 'JJ'), ('bounds', 'NNS'), ('for', 'IN'), ('components', 'NNS'), ('of', 'IN'), ('a', 'DT'), ('minimal', 'JJ'), ('set', 'NN'), ('of', 'IN'), ('solutions', 'NNS'), ('and', 'CC'), ('algorithms', 'NN'), ('of', 'IN'), ('construction', 'NN'), ('of', 'IN'), ('minimal', 'JJ'), ('generating', 'VBG'), ('sets', 'NNS'), ('of', 'IN'), ('solutions', 'NNS'), ('for', 'IN'), ('all', 'DT'), ('types', 'NNS'), ('of', 'IN'), ('systems', 'NNS'), ('are', 'VBP'), ('given', 'VBN'), ('.', '.'), ('these', 'DT'), ('criteria', 'NNS'), ('and', 'CC'), ('the', 'DT'), ('corresponding', 'JJ'), ('algorithms', 'NN'), ('for', 'IN'), ('constructing', 'VBG'), ('a', 'DT'), ('minimal', 'JJ'), ('supporting', 'NN'), ('set', 'NN'), ('of', 'IN'), ('solutions', 'NNS'), ('can', 'MD'), ('be', 'VB'), ('used', 'VBN'), ('in', 'IN'), ('solving', 'VBG'), ('all', 'PDT'), ('the', 'DT'), ('considered', 'VBN'), ('types', 'NNS'), ('of', 'IN'), ('systems', 'NNS'), ('and', 'CC'), ('systems', 'NNS'), ('of', 'IN'), ('mixed', 'JJ'), ('types', 'NNS'), ('.', '.')]


The tokenized text (mainly the nouns and adjectives) is normalized by <b>lemmatization</b>.
In lemmatization different grammatical counterparts of a word will be replaced by single
basic lemma. For example, 'glasses' may be replaced by 'glass'. 

Details about lemmatization: 
    
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html


```python
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

adjective_tags = ['JJ','JJR','JJS']

lemmatized_text = []

for word in POS_tag:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
        
print "Text tokens after lemmatization of adjectives and nouns: \n"
print lemmatized_text
```

    Text tokens after lemmatization of adjectives and nouns: 
    
    ['compatibility', 'of', 'system', 'of', 'linear', 'constraint', 'over', 'the', 'set', 'of', 'natural', 'number', '.', 'criterion', 'of', 'compatibility', 'of', 'a', 'system', 'of', 'linear', 'diophantine', 'equation', ',', 'strict', 'inequations', ',', 'and', 'nonstrict', 'inequations', 'are', 'considered', '.', 'upper', 'bound', 'for', 'component', 'of', 'a', 'minimal', 'set', 'of', 'solution', 'and', 'algorithm', 'of', 'construction', 'of', 'minimal', 'generating', 'set', 'of', 'solution', 'for', 'all', 'type', 'of', 'system', 'are', 'given', '.', 'these', 'criterion', 'and', 'the', 'corresponding', 'algorithm', 'for', 'constructing', 'a', 'minimal', 'supporting', 'set', 'of', 'solution', 'can', 'be', 'used', 'in', 'solving', 'all', 'the', 'considered', 'type', 'of', 'system', 'and', 'system', 'of', 'mixed', 'type', '.']


The <b>lemmatized text</b> is <b>POS tagged</b> here.


```python
POS_tag = nltk.pos_tag(lemmatized_text)

print "Lemmatized text with POS tags: \n"
print POS_tag
```

    Lemmatized text with POS tags: 
    
    [('compatibility', 'NN'), ('of', 'IN'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('constraint', 'NN'), ('over', 'IN'), ('the', 'DT'), ('set', 'NN'), ('of', 'IN'), ('natural', 'JJ'), ('number', 'NN'), ('.', '.'), ('criterion', 'NN'), ('of', 'IN'), ('compatibility', 'NN'), ('of', 'IN'), ('a', 'DT'), ('system', 'NN'), ('of', 'IN'), ('linear', 'JJ'), ('diophantine', 'JJ'), ('equation', 'NN'), (',', ','), ('strict', 'JJ'), ('inequations', 'NNS'), (',', ','), ('and', 'CC'), ('nonstrict', 'JJ'), ('inequations', 'NNS'), ('are', 'VBP'), ('considered', 'VBN'), ('.', '.'), ('upper', 'JJ'), ('bound', 'NN'), ('for', 'IN'), ('component', 'NN'), ('of', 'IN'), ('a', 'DT'), ('minimal', 'JJ'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('and', 'CC'), ('algorithm', 'NN'), ('of', 'IN'), ('construction', 'NN'), ('of', 'IN'), ('minimal', 'JJ'), ('generating', 'VBG'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('for', 'IN'), ('all', 'DT'), ('type', 'NN'), ('of', 'IN'), ('system', 'NN'), ('are', 'VBP'), ('given', 'VBN'), ('.', '.'), ('these', 'DT'), ('criterion', 'NN'), ('and', 'CC'), ('the', 'DT'), ('corresponding', 'JJ'), ('algorithm', 'NN'), ('for', 'IN'), ('constructing', 'VBG'), ('a', 'DT'), ('minimal', 'JJ'), ('supporting', 'NN'), ('set', 'NN'), ('of', 'IN'), ('solution', 'NN'), ('can', 'MD'), ('be', 'VB'), ('used', 'VBN'), ('in', 'IN'), ('solving', 'VBG'), ('all', 'PDT'), ('the', 'DT'), ('considered', 'VBN'), ('type', 'NN'), ('of', 'IN'), ('system', 'NN'), ('and', 'CC'), ('system', 'NN'), ('of', 'IN'), ('mixed', 'JJ'), ('type', 'NN'), ('.', '.')]


Any word from the lemmatized text, which isn't a noun, adjective, or gerund (or a 'foreign word'), is here
considered as a <b>stopword</b> (non-content). This is based on the assumption that usually keywords are noun,
adjectives or gerunds. 

Punctuations are added to the stopword list too.


```python
stopwords = []

wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])

punctuations = list(str(string.punctuation))

stopwords = stopwords + punctuations
```

Even if we remove the aforementioned stopwords, still some extremely common nouns, adjectives or gerunds may
remain which are very bad candidates for being keywords (or part of it). 

An external file constituting a long list of stopwords is loaded and all the words are added with the previous
stopwords to create the final list 'stopwords-plus' which is then converted into a set. 

(Source of stopwords data: https://www.ranks.nl/stopwords)

Stopwords-plus constitute the sum total of all stopwords and potential phrase-delimiters. The contents of this
set will be used to partition the lemmatized text into phrases. 

Phrases should constitute a group of consecutively occuring words that has no member from stopwords_plus in
between. Example: "Neural Network".
    
Each phrase is a <b>keyword candidate</b>. 
    
There are some exceptions, that is, there are some possible cases where a good keyword candidate may contain 
stopword in between. Example: "Word of Mouth". 
    
But, for simplicity's sake I will pretend here that such exceptions do not exist.


```python
stopword_file = open("long_stopwords.txt", "r")
#Source = https://www.ranks.nl/stopwords

lots_of_stopwords = []

for line in stopword_file.readlines():
    lots_of_stopwords.append(str(line.strip()))

stopwords_plus = []
stopwords_plus = stopwords + lots_of_stopwords
stopwords_plus = set(stopwords_plus)

#Stopwords_plus contain total set of all stopwords and phrase delimiters that
#will be used for partitioning the text into phrases (candidate keywords).
```

Phrases are generated by partitioning the lemmatized text using the members of stopwords_plus 
as delimiters.


```python
phrases = []

phrase = " "
for word in lemmatized_text:
    
    if word in stopwords_plus:
        if phrase!= " ":
            phrases.append(str(phrase).split())
        phrase = " "
    elif word not in stopwords_plus:
        phrase+=str(word)
        phrase+=" "

print "Partitioned Phrases: \n"
print phrases
```

    Partitioned Phrases: 
    
    [['compatibility'], ['system'], ['linear', 'constraint'], ['set'], ['natural', 'number'], ['criterion'], ['compatibility'], ['system'], ['linear', 'diophantine', 'equation'], ['strict', 'inequations'], ['nonstrict', 'inequations'], ['upper', 'bound'], ['component'], ['minimal', 'set'], ['solution'], ['algorithm'], ['construction'], ['minimal', 'generating', 'set'], ['solution'], ['type'], ['system'], ['criterion'], ['corresponding', 'algorithm'], ['constructing'], ['minimal', 'supporting', 'set'], ['solution'], ['solving'], ['type'], ['system'], ['system'], ['mixed', 'type']]


Following is the RAKE algorithm.

Frequency of each words in the list of phrases, are calculated here. 

The degree of each words are calculating by adding the length of all the
phrases where the word occurs.

Each word scores are caclulated by dividing degree of the word by its frequency.



```python
from collections import defaultdict
from __future__ import division

frequency = defaultdict(int)
degree = defaultdict(int)
word_score = defaultdict(float)

vocabulary = []

for phrase in phrases:
    for word in phrase:
        frequency[word]+=1
        degree[word]+=len(phrase)
        if word not in vocabulary:
            vocabulary.append(word)
            
for word in vocabulary:
    word_score[word] = degree[word]/frequency[word]

print "Dictionary of degree scores for each words under candidate keywords (phrases): \n"
print degree
print "\nDictionary of frequencies for each words under candidate keywords (phrases): \n"
print frequency
print "\nDictionary of word scores for each words under candidate keywords (phrases): \n"
print word_score
```

    Dictionary of degree scores for each words under candidate keywords (phrases): 
    
    defaultdict(<type 'int'>, {'upper': 2, 'set': 9, 'constructing': 1, 'number': 2, 'solving': 1, 'system': 5, 'compatibility': 2, 'strict': 2, 'criterion': 2, 'type': 4, 'minimal': 8, 'supporting': 3, 'generating': 3, 'corresponding': 2, 'linear': 5, 'diophantine': 3, 'component': 1, 'bound': 2, 'nonstrict': 2, 'inequations': 4, 'natural': 2, 'algorithm': 3, 'constraint': 2, 'equation': 3, 'solution': 3, 'construction': 1, 'mixed': 2})
    
    Dictionary of frequencies for each words under candidate keywords (phrases): 
    
    defaultdict(<type 'int'>, {'upper': 1, 'set': 4, 'constructing': 1, 'number': 1, 'solving': 1, 'system': 5, 'compatibility': 2, 'strict': 1, 'criterion': 2, 'type': 3, 'minimal': 3, 'supporting': 1, 'generating': 1, 'corresponding': 1, 'linear': 2, 'diophantine': 1, 'component': 1, 'bound': 1, 'nonstrict': 1, 'inequations': 2, 'natural': 1, 'algorithm': 2, 'constraint': 1, 'equation': 1, 'solution': 3, 'construction': 1, 'mixed': 1})
    
    Dictionary of word scores for each words under candidate keywords (phrases): 
    
    defaultdict(<type 'float'>, {'upper': 2.0, 'set': 2.25, 'constructing': 1.0, 'number': 2.0, 'solving': 1.0, 'system': 1.0, 'compatibility': 1.0, 'strict': 2.0, 'criterion': 1.0, 'type': 1.3333333333333333, 'minimal': 2.6666666666666665, 'supporting': 3.0, 'generating': 3.0, 'corresponding': 2.0, 'linear': 2.5, 'diophantine': 3.0, 'component': 1.0, 'bound': 2.0, 'nonstrict': 2.0, 'inequations': 2.0, 'natural': 2.0, 'algorithm': 1.5, 'constraint': 2.0, 'equation': 3.0, 'solution': 1.0, 'construction': 1.0, 'mixed': 2.0})


The phrase scores are calculated by adding individual scores of each of the words
which are the members of the phrase. 


```python
import numpy as np

phrase_scores = []
keywords = []
phrase_vocabulary=[]

for phrase in phrases:
    if phrase not in phrase_vocabulary:
        phrase_score=0
        for word in phrase:
            phrase_score+= word_score[word]
        phrase_scores.append(phrase_score)
        phrase_vocabulary.append(phrase)

phrase_vocabulary = []
j=0
for phrase in phrases:
    
    if phrase not in phrase_vocabulary:
        keyword=''
        for word in phrase:
            keyword += str(word)+" "
        phrase_vocabulary.append(phrase)
        keyword = keyword.strip()
        keywords.append(keyword)
    
        print "Score of candidate keyword '"+keywords[j]+"': "+str(phrase_scores[j])
        
        j+=1
```

    Score of candidate keyword 'compatibility': 1.0
    Score of candidate keyword 'system': 1.0
    Score of candidate keyword 'linear constraint': 4.5
    Score of candidate keyword 'set': 2.25
    Score of candidate keyword 'natural number': 4.0
    Score of candidate keyword 'criterion': 1.0
    Score of candidate keyword 'linear diophantine equation': 8.5
    Score of candidate keyword 'strict inequations': 4.0
    Score of candidate keyword 'nonstrict inequations': 4.0
    Score of candidate keyword 'upper bound': 4.0
    Score of candidate keyword 'component': 1.0
    Score of candidate keyword 'minimal set': 4.91666666667
    Score of candidate keyword 'solution': 1.0
    Score of candidate keyword 'algorithm': 1.5
    Score of candidate keyword 'construction': 1.0
    Score of candidate keyword 'minimal generating set': 7.91666666667
    Score of candidate keyword 'type': 1.33333333333
    Score of candidate keyword 'corresponding algorithm': 3.5
    Score of candidate keyword 'constructing': 1.0
    Score of candidate keyword 'minimal supporting set': 7.91666666667
    Score of candidate keyword 'solving': 1.0
    Score of candidate keyword 'mixed type': 3.33333333333


The index of the phrase_scores ndarray is then sorted in descending order in terms of
the score values.
The index corresponds to the location of the concerned phrase in phrases list.
So by getting the sorted order of the index, we also get the sorted order of the phrases.
Each phrase can be considered as a <b>candidate keyword</b>. 
We can then simply choose the top n highest scoring candidate keywords and present them as
the final exctracted keywords for the system. 


```python
sorted_index = np.flip(np.argsort(phrase_scores),0)

keywords_num = 10

print "Keywords:\n"

for i in xrange(0,keywords_num):sa
    print str(keywords[sorted_index[i]])+", ",
```

    Keywords:
    
    linear diophantine equation,  minimal supporting set,  minimal generating set,  minimal set,  linear constraint,  natural number,  upper bound,  nonstrict inequations,  strict inequations,  corresponding algorithm, 


# Input:

Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types.

# Extracted Keywords:

* linear diophantine equation,  
* minimal generating set,  
* minimal supporting set,  
* minimal set,  
* linear constraint,  
* natural number,     
* upper bound,
* nonstrict inequations
* strict equations
