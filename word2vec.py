import torch
from random import shuffle
from collections import Counter
import argparse
import random
import time
import numpy as np

def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)

def get_activated_node(len_corpus, sampling_num, prob_table, correct_idx):
    activated_node_lst = [correct_idx]
    lotto_num = random.randint(0, len_corpus - 1)
    for i in range(sampling_num):
        while lotto_num in activated_node_lst:    
            lotto_num = random.randint(0, len_corpus - 1)
        activated_node_lst.append(int(prob_table[lotto_num]))
        lotto_num = random.randint(0, len_corpus - 1)
    return activated_node_lst

def get_word_ngram(word, n):
    ngram_features = []
    length = len(word)
    if length > n:
        for i in range(length-n+1):
            subword = ''.join(word[i:i+n])
            ngram_features.append(subword)
    return ngram_features

def fnv(s, bucket):
    prime = 0x100000001b3
    basis = 0xcbf29ce484222325
    hashing = basis
    for cash in s:
        hashing = hashing^ord(cash)
        hashing = (hashing * prime) % (2**64)
    return hashing % bucket

def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix, update_system, feed_dict):
    activated_node_lst =  feed_dict['activated_node_lst']
    hashedngramsInds = feed_dict['hashedngramsInds']

    sum_of_center_word_grams_vector = torch.sum(inputMatrix[hashedngramsInds, :],dim=0,keepdim=True) #1,D

    score_vector = torch.matmul(sum_of_center_word_grams_vector, torch.t(outputMatrix)) # (1,D) * (D,K) = (1,K)
    score_vector = torch.t(score_vector)

    e = np.exp(score_vector) 
    sig_vector = e/(e+1)

    loss = 0.0
        
    for i, idx in enumerate(activated_node_lst):
        #print(idx)
        if idx == contextWord:
            context_idx = i
            loss -= np.log(sig_vector[i])
        else:
            loss -= np.log(1 - sig_vector[i])

    sig_grad = sig_vector #(K,1)
    sig_grad[context_idx] -= 1

    grad_out = torch.matmul(sig_grad, sum_of_center_word_grams_vector) #(K,1) * (1,D) = (K,D)
    grad_emb = torch.matmul(torch.t(sig_grad), outputMatrix) #(1,K) * (K,D) = (1,D)
    return loss, grad_emb, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):

    sum_of_context_words_vector = torch.sum(inputMatrix[contextWords, :],dim=0,keepdim=True) #1,D

    score_vector = torch.matmul(sum_of_context_words_vector, torch.t(outputMatrix)) # (1,D) * (D,V) = (1,V)
    
    e = torch.exp(score_vector) 
    softmax = e / (torch.sum(e, dim=1, keepdim=True)) #1,V

    loss = -torch.log(softmax[:,centerWord])
    
    #get grad
    softmax_grad = softmax
    softmax_grad[:,centerWord] -= 1.0

    grad_out = torch.matmul(torch.t(softmax_grad), sum_of_context_words_vector) #(1,V) * (1,D) = (V,D)
    grad_emb = torch.matmul(softmax_grad, outputMatrix) #(1,V) * (V,D) = (1,D)
    
    return loss, grad_emb, grad_out


def word2vec_trainer(corpus, word2ind, gram2hashed, ind2hashed, hashed2ind, feed_dict, n_lst, mode, update_system, subsampling, dimension, learning_rate, iteration):
    print("size of corpus: %d" % len(corpus))
    #Only once
    if subsampling is True:
        print("Start SubSampling...")
        prob_of_ss = feed_dict['prob_of_ss']
        destiny = np.random.random(size=len(corpus))

        subsampling_word = []
        for idx, word in enumerate(corpus):
            if destiny[idx] < prob_of_ss[word]:
                subsampling_word.append(idx) 
            else:
                pass
            
        corpus = list(np.delete(corpus,subsampling_word))
        print("Finish SubSampling...")
        print("size of corpus(after): %d" % len(corpus))

    # Xavier initialization of weight matrices
    #W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5) 
    W_emb = torch.randn(len(word2ind)+ len(hashed2ind), dimension) / (dimension**0.5) 
    W_out = torch.randn(len(word2ind), dimension) / (dimension**0.5)  
    window_size = 5

    prob_table = feed_dict['prob_table']
    sum_of_pow_freq = feed_dict['sum_of_pow_freq'] 

    sampling_num = 5
    losses=[]
    for i in range(iteration):
        learning_rate = 0.025 * (1-i/iteration)
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        af_centerword = "<"+centerword+">"
        centerInd =  word2ind[af_centerword]

        contextInds = [word2ind["<"+i+">"] for i in context]

        #n-gram
        centerword_ngrams = []
        for n in n_lst:
            centerword_ngrams += get_word_ngram(af_centerword, n)
        hashedngramsInds = [hashed2ind[gram2hashed[ngram]] for ngram in centerword_ngrams]
        hashedngramsInds.append(centerInd)

        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb[contextInds] -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())

        elif mode=="SG":
            for contextInd in contextInds:
                feed_dict2={}

                activated_node_lst = get_activated_node(sum_of_pow_freq, sampling_num, prob_table, contextInd)
    
                #add centerInd
                feed_dict2['hashedngramsInds'] = hashedngramsInds
                feed_dict2['activated_node_lst'] = activated_node_lst

                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out[activated_node_lst], update_system, feed_dict2)
                W_emb[hashedngramsInds] -= learning_rate*G_emb.squeeze()
                W_out[activated_node_lst] -= learning_rate*G_out
                losses.append(L.item())
        else:
            print("Unkwnown mode : "+mode)
            exit()

        if i%10000==0:
            avg_loss=sum(losses)/len(losses)
            print("i: %d, Loss : %f, learning_rate : %f " %(i, avg_loss, learning_rate))
            losses=[]

    return W_emb, W_out

def sim(testword, word2ind, ind2word,all_emb, word_emb, gram2hashed, hashed2ind):
    length = (word_emb*word_emb).sum(1)**0.5

    temp = "<"+testword+">"
    testword_ngrams = []
    n_lst=[3,4,5,6]
    for n in n_lst:
        testword_ngrams += get_word_ngram(temp, n)

    hashedngramsInds= []
    for ngram in testword_ngrams:
        if ngram in gram2hashed.keys():
            hashedngramsInds.append(hashed2ind[gram2hashed[ngram]])

    if temp in word2ind.keys():
        wi = word2ind[temp]
        hashedngramsInds.append(wi)
    
    sum_of_center_word_grams_vector = torch.sum(all_emb[hashedngramsInds, :],dim=0,keepdim=True) #1,D

    inputVector_length = (sum_of_center_word_grams_vector*sum_of_center_word_grams_vector).sum(1)**0.5
    inputVector = sum_of_center_word_grams_vector.reshape(1,-1)/inputVector_length

    sim = (inputVector@word_emb.t())[0]/length
    values, indices = sim.squeeze().topk(10)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('us', metavar='us', type=str, help="NS for Negative Sampling only")
    parser.add_argument('ss', metavar='ss', type=bool, help="True or False")
    
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    update_system = args.us
    subsampling = args.ss

	#Load and tokenize corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("tokenizing...")
    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)

    frequency = Counter(processed)
    vocabulary = set(processed)

    #Assign an index number to a word
    word2ind = {}
    word2ind["< >"]=0
    i = 1
    for word in vocabulary:
        #print(word)
        word2ind["<"+word+">"] = i
        i+=1
    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Word Vocabulary size")
    print(len(word2ind))
    print()
    feed_dict = {}
    '''
    Sub_Sampling or Not
    '''
    if subsampling == True:
        sum_of_freq = 0
        for word, freq in frequency.items():
            sum_of_freq += freq       

        ratio_of_freq = {}
        for word, freq in frequency.items():
            ratio_of_freq[word] = freq / sum_of_freq
        
        t = 10e-5

        prob_of_ss = {}
        for word, freq in frequency.items():
            prob_of_ss[word] = 1- (np.sqrt(t / ratio_of_freq[word])) #P(w_i) , discard frequent words with prob

        feed_dict['prob_of_ss'] = prob_of_ss  

    '''
    For Negative Sampling
    '''
    pow_of_freq = {}
    for word, freq in frequency.items():
        pow_of_freq[word] = freq ** 0.75
    
    sum_of_pow_freq = 0
    for freq in pow_of_freq.values():
        sum_of_pow_freq += freq
    
    idx = 0
    prob_table = np.zeros(int(sum_of_pow_freq))
    for word, freq in pow_of_freq.items():
        freq = int(freq)
        prob_table[idx:idx+freq] = word2ind["<"+word+">"]
        idx = idx+freq 
    
    feed_dict['prob_table'] = prob_table
    feed_dict['sum_of_pow_freq']= int(sum_of_pow_freq) 
    
    print("making n-gram ....")
    #make n-gram
    gram2hashed = {}
    ind2hashed = {}
    hashed2ind = {}
    start = time.time()
    n_lst=[3,4,5,6]
    s = 0
    for word in word2ind:
        ngram_lst = []
        for n in n_lst:
            ngram_lst += get_word_ngram(word,n)
        s+=len(ngram_lst)
        for ngram in ngram_lst:
            hashed = fnv(ngram, 2100000)
            gram2hashed[ngram]= hashed
            if hashed not in hashed2ind.keys():
                ind2hashed[i]=hashed
                hashed2ind[hashed]= i
                i += 1
    print("s: {}".format(s))
    print("consume time: {}".format(time.time()-start))
    print("{}grams hashed size: {}".format(n_lst, len(hashed2ind)))
    print("Total WordEmbedding size: {}".format(len(word2ind) + len(hashed2ind)))
    print()
    
    #Training section
    start = time.time()
    emb,_ = word2vec_trainer(processed, word2ind, gram2hashed, ind2hashed, hashed2ind, feed_dict, n_lst, mode, update_system, subsampling, dimension=300, learning_rate=0.025, iteration=2000)
    
    print("Training Consume time: {}".format(time.time()-start))
    
    #make word_emb
    start = time.time()
    word_emb = torch.zeros(len(word2ind), 300)
    for word in word2ind:
        idx = word2ind[word]

        testword_ngrams = []
        for n in n_lst:
            testword_ngrams += get_word_ngram(word, n)

        hashedngramsInds= []
        for ngram in testword_ngrams:
            if ngram in gram2hashed.keys():
                hashedngramsInds.append(hashed2ind[gram2hashed[ngram]])

        hashedngramsInds.append(idx)
    
        sum_of_center_word_grams_vector = torch.sum(emb[hashedngramsInds, :],dim=0,keepdim=True) #1,D
        word_emb[idx]=sum_of_center_word_grams_vector

    print("Making word_emb Consume time: {}".format(time.time()-start))

    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after", "narrow-mindedness", "department", "campfires", "knowing", "urbanize", "imperfection", "principality", "abnormal", "secondary", "ungraceful"]
    #testwords = ["narrow-mindedness", "department", "campfires", "knowing", "urbanize", "imperfection", "principality", "abnormal", "secondary", "ungraceful"]
    for tw in testwords:
        sim(tw,word2ind,ind2word,emb,word_emb, gram2hashed, hashed2ind)

main()