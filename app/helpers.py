#from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
from semantic import *

from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import unicodedata
import gensim.models.word2vec as w2v
import multiprocessing

from nltk.tag import StanfordPOSTagger, StanfordNERTagger
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import Counter
from itertools import chain
import unicodedata
import pandas as pd
import numpy as np
import string
import pickle
import re
import pandas
import nltk
import copy

def get_answer_pos(tagged_question, tagged_sentence, answer):
    answer_lenth = len(answer.split())
    indexes = [i for i, t in enumerate(tagged_question) if (t[0] == '' or t[0] == '_' or t[0] == "'_")]
    starting_index = indexes[0]
    ending_index = starting_index+answer_lenth-1
    return [tag[1] for tag in tagged_sentence[starting_index:ending_index+1]]

def get_answer_depth(question):
    return question.split().index('_')/len(question.split())

def get_context_pos(tagged_question, tagged_sentence, answer):
    answer_lenth = len(answer.split())
    indexes = [i for i, t in enumerate(tagged_question) if (t[0] == '' or t[0] == '_' or t[0] == "'_")]
    starting_index = indexes[0]
    ending_index = starting_index+answer_lenth-1
    
    if starting_index == 0:
        context = ('XX', 'XX', tagged_sentence[ending_index+1], tagged_sentence[ending_index+2])
    elif starting_index == 1:
        context = ('XX', tagged_sentence[starting_index-1], tagged_sentence[ending_index+1], tagged_sentence[ending_index+2])
    elif ending_index == len(tagged_sentence)-2:
        context = (tagged_sentence[starting_index-2], tagged_sentence[starting_index-1], tagged_sentence[ending_index+1], 'XX')
    elif ending_index == len(tagged_sentence)-1:
        context = (tagged_sentence[starting_index-2], tagged_sentence[starting_index-1], 'XX', 'XX')
    else:
        context = (tagged_sentence[starting_index-2], tagged_sentence[starting_index-1], tagged_sentence[ending_index+1], tagged_sentence[ending_index+2])

    return [t[1] for t in context]

def get_answer_word_count(sentence, answer):
    word_count = Counter(sentence.split())
    return [word_count[word] for word in answer.split()]

def get_metadata(sentence, tagged_sentence, question, tagged_question, answer):
    answer_pos = get_answer_pos(tagged_question, tagged_sentence, answer)
    word_count = get_answer_word_count(sentence, answer)
    context_pos = get_context_pos(tagged_question, tagged_sentence, answer)
    answer_depth = get_answer_depth(question)
    answer_length = len(answer.split())
    return ({'answer_pos': answer_pos[0:5], 'word_count': word_count[0:5], 'context_pos': context_pos, 'answer_depth': answer_depth, 'answer_length': answer_length})

def answer_pos_vectorizer(metadata):
    answer_pos = copy.deepcopy(metadata)['answer_pos']
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PUNCT']
    
    if len(answer_pos) < 5:
        slots_short = 5-len(answer_pos)
        for _ in range(slots_short):
            answer_pos.append('XX')
    
    vectors = [[1 if answer_pos[index] == tag else 0 for tag in pos_tags] for index in range(5)]
    return np.array(vectors).flatten()

def word_count_vectorizer(metadata):
    word_count = metadata['word_count']
    
    if len(word_count) < 5:
        slots_short = 5-len(word_count)
        for _ in range(slots_short):
            word_count.append(0)
    
    return np.array(word_count)

def context_pos_vectorizer(metadata):
    context_pos = copy.deepcopy(metadata)['context_pos']
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PUNCT']

    vectors = [[1 if context_pos[index] == tag else 0 for tag in pos_tags] for index in range(4)]
    return np.array(vectors).flatten()

def sentence_vectorizer(metadata):
    v = chain([metadata['answer_depth']],
              [metadata['answer_length']],
              word_count_vectorizer(metadata),
              answer_pos_vectorizer(metadata),
              context_pos_vectorizer(metadata)
             )
    return np.array(list(v))

def tag_sentence(sentence, tagger='standard'):
    wt = word_tokenize(sentence)
    if tagger == 'stanford':
        jar = '/Users/alex/stanford-postagger-2017-06-09/stanford-postagger.jar'
        model = '/Users/alex/stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
        st = StanfordPOSTagger(model, jar)
        return st.tag(wt)
    else:
        pos = pos_tag(wt)
        return pos

def get_trees(sentence, patterns):
    tagged_sentence = tag_sentence(sentence)
    forest = []
    for pattern in patterns:
        cp = nltk.RegexpParser(pattern)
        tree = cp.parse(tagged_sentence)
        forest.append((tree, pattern))
    return (tagged_sentence, forest)

def blankify(tree_list):
    indexes = [i for i, t in enumerate(tree_list) if type(t[0]) == tuple]
    all_questions = []
    for index in indexes:
        question = []
        tagged_question = []
        answer = []
        for i, word in enumerate(tree_list):
            if i == index:
                for el in word:
                    question.append('_')
                    answer.append(el[0])
                    tagged_question.append(('_', el[1]))
            elif i in indexes and i != index:
                for el in word:
                    question.append(el[0])
                    tagged_question.append(el)
            else:
                question.append(word[0])
                tagged_question.append(word)
        all_questions.append({'question':" ".join(question),'tagged_question':tagged_question, 'answer':" ".join(answer)})
    return all_questions

def make_all_questions(sentence, patterns):
    tagged_sentence, all_matches = get_trees(sentence, patterns)
    all_question_combos = []
    for tree in all_matches:
        list_tree = [list(b)for b in tree[0]]
        questions = blankify(list_tree)

        for question in questions:
            question.update({'pattern': tree[1]})
        all_question_combos.append(questions)

    all_questions_faltten = []
    for combo in all_question_combos:
        if len(combo) == 1:
            combo[0].update({'sentence': sentence, 'tagged_sentence': tagged_sentence})
            all_questions_faltten.append(combo[0])
        if len(combo) > 1:
            for q in combo:
                q.update({'sentence': sentence, 'tagged_sentence': tagged_sentence})
                all_questions_faltten.append(q)
    return all_questions_faltten

def get_top_patterns(num=25):
    patterns = ['1: {<NNP>}', '2: {<DT><NN>}', '3: {<NN>}', '4: {<DT><JJ><NN>}', '5: {<NNS>}',
     '6: {<CD>}', '7: {<JJ><NN>}', '8: {<JJ><NNS>}', '9: {<DT><NNP>}', '10: {<NNP><NNP>}',
     '11: {<DT><NN><NN>}', '12: {<VBN>}', '13: {<VBD>}', '14: {<DT><NNP><NNP>}',
     '15: {<VBG>}', '16: {<DT><NNS>}', '17: {<VB>}', '18: {<JJ>}', '19: {<NN><NNS>}',
     '20: {<NN><NN>}', '21: {<JJ><NNP>}', '22: {<PRP>}', '23: {<DT><NNP><NN>}',
     '24: {<DT><NNP><NNP><NNP>}', '25: {<NNP><NNP><NNP>}']
    return patterns[1:num]

def predict_best_question(questions, model, top_n=1):
    max_pred = 0
    predictions = []
    for i, q in enumerate(questions):
        question_vector = Sentence(q['sentence'], q['question'], q['answer']).vector()
        pred = model.predict_proba(question_vector.reshape(1, -1))[0][1]
        predictions.append((q, pred))
        top_questions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return top_questions

def get_best_sentences(text, num=1):
    sentence_count = num
    parser = PlaintextParser(text, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')
    
    return summarizer(parser.document, sentence_count)

def unpunkt(text):
    return "".join([c if unicodedata.category(c)[0] != 'P' else ' ' for c in text])

def get_word2vec(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(text)
    clean_sentences = []

    for sentence in raw_sentences:
        unpunkt = "".join([c if unicodedata.category(c)[0] != 'P' else '' for c in sentence ])
        clean_sentences.append(nltk.word_tokenize(unpunkt))

    num_features = 200
    min_word_count = 1
    num_workers = multiprocessing.cpu_count()
    context_size = 3
    downsampling = 1e-3

    text2vec = w2v.Word2Vec(sg=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context_size, sample=downsampling)

    text2vec.build_vocab(clean_sentences)
    text2vec.train(clean_sentences, total_examples=text2vec.corpus_count, epochs=text2vec.iter)

    return text2vec

def make_pattern_distractors(text, pattern, answer):
    tagged_text = pos_tag(word_tokenize(text))
    cp = nltk.RegexpParser(pattern)
    tree = cp.parse(tagged_text)
    indexes = [i for i, t in enumerate(tree) if type(t) == nltk.tree.Tree]
    pattern_matches = []

    for index in indexes:
        tag = [t for t in tree[index]]
        if tag not in pattern_matches:
            pattern_matches.append(tag)

    #print(pattern_matches)
    choices = []

    for match in pattern_matches:
        score = get_distractor_similarity(text, pos_tag(word_tokenize(answer)), match)
        phrase = " ".join(t[0] for t in match)
        choices.append((phrase, score))

    
    if 'JJ' in pattern:
        try:
            index = [i for i, t in enumerate(pos_tag(word_tokenize(answer))) if t[1] == 'JJ'][0]
            adj = answer.split()[index]
            adj_syn = wn.synsets(adj, pos='a')[0]
            antonym = adj_syn.lemmas()[0].antonyms()[0].name()
            a = answer.split()
            a[index] = antonym
            whole_antonym = " ".join(a)
        except:
            whole_antonym = None
        
        if whole_antonym != None:
            choices.append((whole_antonym, 100))
        
    return sorted(choices, key=lambda x: x[1], reverse=True)

def ner_chunker(ner):
    chunked_ner = []
    for i, tag in enumerate(ner):
        if i == 0:
            chunked_ner.append(list(tag))
        elif tag[1] != 'O' and tag[1] == chunked_ner[-1][1]:
            chunked_ner[-1][0] += " "+tag[0]
        else:
            chunked_ner.append(list(tag))
    return chunked_ner

def make_ner_dict(chunked_ner):
    ner_dict = {}
    for tag in chunked_ner:
        if tag[1] != 'O':
            if tag[1] in ner_dict:
                ner_dict[tag[1]].append(tag[0])
            else:
                ner_dict[tag[1]] = [tag[0]]
    return ner_dict

def get_answer_ner(answer):
    ner = get_ner7(answer)
    return [tag[1] for tag in ner][0]

def get_ner_distractors(text, answer):
    answer_ner = get_answer_ner(answer)
    if answer_ner != 'O':
        ner = get_ner7(text)
        chunked_ner = ner_chunker(ner)
        ner_dict = make_ner_dict(chunked_ner)
        alternatives = set([alt for alt in ner_dict[answer_ner] if answer not in alt])
    else:
        alternatives = []
    return set(alternatives)

def get_distractor_similarity(text, tagged_answer, tagged_distractor):
    # answers and distractors will all be the same length and have the same POS pattern
    pos_dict = {
        'NNP': 'n',
        'NN': 'n',
        'NNS': 'n',
        'JJ': 'a',
        'VBD': 'v',
        'VB': 'v',
    }
    
    noun_indexes = [i for i, tup in enumerate(tagged_answer) if pos_dict[tup[1]] == 'n']
    
    cumulative_score = 0
    for index in noun_indexes:
        answer_word = tagged_answer[index][0]
        #answer_syn = lesk(word_tokenize(text), answer_word, 'n')
        answer_syn = wn.synsets(answer_word, pos='n')[0]
        for i in noun_indexes:
            try:
                distractor_word = tagged_distractor[i][0]
                #distractor_syn = lesk(word_tokenize(text), distractor_word, 'n')
                distractor_syn = wn.synsets(distractor_word, pos='n')[0]
                score = wn.lch_similarity(answer_syn, distractor_syn)
                if score == None:
                    cumulative_score += 0
                else:
                    cumulative_score += score
            except:
                print("Error", tagged_distractor)
        
        
    return cumulative_score/len(noun_indexes)**2

def make_distractors(text, pattern, answer):
    # 1. check for NERs
    # 2. check for noun-noun similarities within the document
    # 3. generate distractors from synset hyper/hypo nyms

    # check for NERs
    ner_distractors = get_ner_alternatives(text, answer)[0]
    
    if len(ner_distractors) > 0:
        return ner_distractors[:3] # this won't break if there aren't 3 records!
    else:
        pattern_distractors = make_pattern_distractors(text, pattern, answer)

    return pattern_distractors[1:5]

def get_ner_alternatives(text, answer):
    answer_ner = get_answer_ner(answer)
    if answer_ner != 'O':
        ner = get_ner7(text)
        chunked_ner = ner_chunker(ner)
        ner_dict = make_ner_dict(chunked_ner)
        alternatives = set([alt for alt in ner_dict[answer_ner] if answer not in alt])
    else:
        alternatives = []
    return [list(alternatives), answer_ner]

def get_ner7(text):
    wt = word_tokenize(text)
    ner_model_seven = '/Users/alex/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz'
    ner_jar = '/Users/alex/stanford-ner-2017-06-09/stanford-ner.jar'
    st_ner7 = StanfordNERTagger(ner_model_seven, ner_jar)
    ner7 = st_ner7.tag(wt)
    
    return ner7