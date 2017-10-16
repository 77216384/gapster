from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from nltk.tag import StanfordPOSTagger, StanfordNERTagger
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
        forest.append(tree)
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
        list_tree = [list(b)for b in tree]
        questions = blankify(list_tree)
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
    return patterns[0:num]

def predict_best_question(questions, model):
    max_pred = 0
    best_q_index = 0
    for i, q in enumerate(questions):
        meta = get_metadata(q['sentence'], q['tagged_sentence'], q['question'], q['tagged_question'], q['answer'])
        sv = sentence_vectorizer(meta)
        pred = model.predict_proba(sv.reshape(1, -1))[0][2]
        if pred > max_pred:
            max_pred = pred
            best_q_index = i
    return (questions[best_q_index], max_pred)

def get_best_sentences(text, num=1):
    sentence_count = num
    parser = PlaintextParser(text, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')
    
    return summarizer(parser.document, sentence_count)
