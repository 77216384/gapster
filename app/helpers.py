#from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
from __future__ import unicode_literals, division, print_function

from semantic import *

from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import unicodedata
from practnlptools.tools import Annotator

from collections import Counter
from itertools import chain
import unicodedata
import pandas as pd
import numpy as np
import string
import pickle
import spacy
import re
import pandas
import nltk
import copy

def get_matching_ents(spacy_answer, spacy_text):
    return [ent for ent in spacy_text.ents if ent[0].ent_type_ == spacy_answer[0].ent_type_]

def check_question_quality(question, spacy_text):
    if question['spacy_answer'][0].ent_type_ != '':
        matching_ents = get_matching_ents(spacy_answer, spacy_text)
        if len(matching_ents) >= 3:
            question.update({'quality':True})
            question.update({'matching_ents': matching_ents})
        else:
            question.update({'quality':False})
            question.update({'matching_ents': matching_ents})
    else:
        question.update({'quality':True})
        question.update({'matching_ents': []})

    return question


def predict_best_question(questions, model, nlp, top_n=1):
    max_pred = 0
    predictions = []
    for i, q in enumerate(questions):
        question_vector = SemanticSentence(q['sentence'], q['question'], q['answer'], nlp, srl=q['srl']).vector()
        pred = model.predict_proba(question_vector.reshape(1, -1))[0][1]
        predictions.append((q, pred))
    top_questions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    #remove duplicate sentences
    non_dup_top_questions = []
    for tq in top_questions:
        if tq[0]['sentence'] not in [q[0]['sentence'] for q in non_dup_top_questions]:
            non_dup_top_questions.append(tq)

    return [q[0] for q in non_dup_top_questions[:top_n]]

def get_best_sentences(text, num=1):
    sentence_count = num
    parser = PlaintextParser(text, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')
    
    return [unicode(s) for s in summarizer(parser.document, sentence_count)]

def unpunkt(text):
    return "".join([c if unicodedata.category(c)[0] != 'P' else ' ' for c in text])