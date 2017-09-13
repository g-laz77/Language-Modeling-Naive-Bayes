# Language-Modeling-Naive-Bayes
A language modelling of subreddits for NLP course at IIIT-H

## Tokenisation
------------------
- [x] The data contains some challenging aspects for tokenisation. Observe the data and include them in the report. Implement a tokeniser which can handle these problems. Mention your design choices and how your algorithm handles these problems. 

## Language Modeling
----------------------------
- [x] Implement unigram, bigram and trigram language models. 
- [x] Plot log-log curve and zipf curve for the above
- [x] Implement laplace smoothing. Compare the effect of smoothing on different values for V (200, 2000, current size of vocabulary, 10*size of vocabulary). Plot these to compare.
- [ ] Implement Witten-Bell backoff. 
- [ ] Implement Kneser-Ney smoothing. 
- [ ] Compare the effects of the three smoothing techniques. (Plot)
- [ ] In Kneser-Ney, what happens if we use the estimates from laplace and wittenbell in the absolute discounting step ?. (Plot & Compare)
- [ ] Using KN-estimates from the three sources, generate text with unigram, bigram and trigram probabilities. 

## Naive Bayes
------------------
- [ ] Plot the zipf's curves of all the three sources on one graph. Where do they match ? Where don't they match ?
- [ ] Formulate tokenisation as a supervised problem. Annotate a small section of each source. Use the language models you have implemented.  Implement naive bayes algorithm for this problem.
- [ ] How does it perform ? .
