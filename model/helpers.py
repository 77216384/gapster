from collections import Counter
from itertools import chain
import unicodedata
import numpy as np
import string


def get_answer_pos(tagged_sentence, answer):
    pos_dict = {t[0]:t[1] for t in tagged_sentence}
    answer_list = answer.split()
    return [pos_dict[w] for w in answer_list]


def get_answer_depth(sentence, answer):
    return sentence.index(answer)/len(sentence)


def get_context_pos(tagged_question, tagged_sentence, answer):
    answer_lenth = len(answer.split())
    indexes = [i for i, t in enumerate(tagged_question) if (t[0] == '' or t[0] == '_')]
    starting_index = indexes[0]
    ending_index = starting_index+answer_lenth-1
    
    if starting_index == 0:
        context = ('XX', 'XX', tagged_sentence[indexes[-1]+1], tagged_sentence[indexes[-1]+2])
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
    answer_pos = get_answer_pos(tagged_sentence, answer)
    word_count = get_answer_word_count(sentence, answer)
    context_pos = get_context_pos(tagged_question, tagged_sentence, answer)
    answer_depth = get_answer_depth(sentence, answer)
    answer_length = len(answer.split())
    return ({'answer_pos': answer_pos[0:5], 'word_count': word_count[0:5], 'context_pos': context_pos, 'answer_depth': answer_depth, 'answer_length': answer_length})


def answer_pos_vectorizer(metadata):
    answer_pos = metadata['answer_pos']
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
    context_pos = metadata['context_pos']
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
