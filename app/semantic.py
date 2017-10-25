from __future__ import unicode_literals, division, print_function

import practnlptools
from practnlptools.tools import Annotator
from collections import Counter
import pandas as pd
import numpy as np
import unicodedata
import pickle
import spacy
import nltk
import re

#nlp = spacy.load('en_core_web_md')

class DistractorSet(object):
    def __init__(self, text, answer, nlp):
        self.nlp = nlp
        self.answer = answer
        self.raw_text = text
        self.spacy = self.nlp(text)
        self.distractors = []
        self.matching_ents = self.get_matching_ents()
        self.noun_chunks = list(self.spacy.noun_chunks)
        self.pos_pattern_matches = self.get_pos_pattern_matches()

    def get_matching_ents(self):
        return [ent for ent in self.spacy.ents if ent.ent_id == self.answer.ent_id]

    def get_pos_pattern_matches(self):
        pos_tag_pattern = [w.tag_ for w in self.answer]
        matcher = spacy.matcher.Matcher(self.nlp.vocab)
        matcher.add_pattern("answer_pattern", [{spacy.attrs.TAG: tag} for tag in pos_tag_pattern])
        matches = matcher(self.spacy)
        return [self.spacy[m[2]:m[3]] for m in matches]

    def collect_distrctors(self):
        return self.matching_ents + self.noun_chunks + self.pos_pattern_matches

    def remove_answer_overlappers(self, distractors):
        return [d for d in distractors if d.text not in answer.text]

    def get_similarities_to_answer(self, distractors):
        return [(d, answer.similarity(d)) for d in distractors]

    def sort_distractors(self, distractors):
        return sorted(distractors, key=lambda x: x[1], reverse=True)

    def filter_duplicates(self, sorted_distractors):
        non_duplicates = set()
        for sd in sorted_distractors:
            if sd.text not in {nd[0].text for nd in non_duplicates}:
                non_duplicates.add(sd)
        return non_duplicates

    def filter_subsets(self, sorted_distractors):
        non_subsets = set()
        for sd in sorted_distractors:
            if sd[0].text not in " ".join({ns[0].text for ns in non_subsets}):
                non_subsets.add(sd)
        return non_subsets

    def filter_root_duplicates(self, sorted_distractors):
        non_root_duplicates = set()
        for sd in sorted_distractors:
            if sd[0].root.text not in {nrd[0].root.text for nrd in non_root_duplicates}:
                non_root_duplicates.add(sd)
        return non_root_duplicates

    def make_distractors(self, answer):
        #all_distractors = []

        #matching_ents = [ent for ent in self.spacy.ents if ent.ent_id == answer.ent_id]

        #for ent in matching_ents:
            #all_distractors.append(ent)
        #for chunk in self.spacy.noun_chunks:
            #all_distractors.append(chunk)

        # make a Matcher for POS tags of answer
        #pos_tags = [w.tag_ for w in answer]
        #matcher = spacy.matcher.Matcher(self.nlp.vocab)
        #matcher.add_pattern("answer_pattern", [{spacy.attrs.TAG: tag} for tag in pos_tags])
        #matches = matcher(self.spacy)
        #for m in matches:
        #    phrase = self.spacy[m[2]:m[3]]
        #    all_distractors.append(phrase)

        #code something here which removes items that are entirely contined in the answer
        #non_overlapping_distractors = [d for d in all_distractors if d.text not in answer.text]

        #distractors = [(d, answer.similarity(d)) for d in non_overlapping_distractors]

        #keep duplicate with highest similarity and discard the rest
        #sorted_distractors = sorted(distractors, key=lambda x: x[1], reverse=True)
        
        #sorted_distractors = sorted_distractors[:15]

        #non_dup_distractors = []
        #for sd in sorted_distractors:
        #    if sd[0].text not in [td[0].text for td in non_dup_distractors]:
        #        non_dup_distractors.append(sd)

        non_subset_distractors = []
        #for i, td in enumerate(non_dup_distractors):
        #    if i == 0:
        #        non_subset_distractors.append(td)
        #    elif td[0].text not in " ".join([x[0].text for x in non_subset_distractors]):
        #        non_subset_distractors.append(td)

        #root_unique_distractors = []
        #for i, td in enumerate(non_subset_distractors):
        #    if i == 0:
        #        root_unique_distractors.append(td)
        #    elif not td[0].root.text in [x[0].root.text for x in root_unique_distractors]:
        #        root_unique_distractors.append(td)

        #return [(rud[0].text, rud[1]) for rud in root_unique_distractors[:3]]

        candidate_distractors = self.collect_distrctors()
        distractors = self.remove_answer_overlappers(distractors)
        distractors = self.get_similarities_to_answer(distractors)
        sorted_distractors = self.sorted_distractors(distractors)
        sorted_distractors = sorted_distractors[:15]
        sorted_distractors = self.filter_duplicates(sorted_distractors)
        sorted_distractors = self.filter_subsets(sorted_distractors)
        sorted_distractors = self.filter_root_duplicates(sorted_distractors)

        return [sd[0].text, sd[1] for sd in sorted_distractors]


class SemanticSentence(object):
    def __init__(self, sentence, question, answer, nlp, srl=None):
        if srl == None:
            self.ascii_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
            self.ascii_question = unicodedata.normalize('NFKD', question).encode('ascii','ignore')
            self.ascii_answer = unicodedata.normalize('NFKD', answer).encode('ascii','ignore')
            self.annotator=Annotator()
            self.srl = self.annotator.getAnnotations(self.ascii_sentence)['srl']
            self.answer_srl_label = self.set_answer_srl_label()
        else:
            self.srl = srl

        self.nlp = nlp
        self.raw_sentence = sentence
        self.raw_question = question
        self.raw_answer = answer
        #self.ascii_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
        #self.ascii_question = unicodedata.normalize('NFKD', question).encode('ascii','ignore')
        #self.ascii_answer = unicodedata.normalize('NFKD', answer).encode('ascii','ignore')
        self.spacy_sent = self.nlp(self.raw_sentence)
        self.spacy_ques = self.nlp(self.raw_question)
        self.answer_length = self.set_answer_length()
        self.spacy_answer = self.set_spacy_answer()
        #self.annotator=Annotator()
        #self.srl = self.annotator.getAnnotations(self.ascii_sentence)['srl']
        self.answer_pos = self.set_answer_pos()
        self.answer_ner = self.set_answer_ner()
        self.answer_ner_iob  = self.set_answer_ner_iob()
        #self.answer_srl_label = self.set_answer_srl_label()
        self.answer_depth = self.set_answer_depth()
        self.answer_word_count = self.set_answer_word_count()
        self.all_pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'PUNCT']
        self.all_ner_tags = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        self.all_srl_labels = ['V', 'A0', 'A1', 'A2', 'C-arg', 'R-arg', 'AM-ADV', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-PRP', 'AM-REC', 'AM-TMP']
        
    def set_answer_length(self):
        return len(self.raw_answer.split())
    
    def set_answer_depth(self):
        try:
            return self.raw_question.split().index('_')/len(self.raw_question.split())
        except:
            return self.raw_question.index('_')/len(self.raw_sentence)
        
    def set_answer_pos(self):
        output = [word.tag_ for word in self.spacy_answer][:5]
        return self.take5(output)
        
    def set_spacy_answer(self):
        self.answer_index_start = [i for i, word in enumerate(self.spacy_ques) if word.text == '_'][0]
        self.answer_index_end = (self.answer_index_start + self.answer_length) - 1
        return [word for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]]
        
    def set_answer_word_count(self):
        self.word_count = Counter([word.lemma_ for word in self.spacy_sent])
        output =  [self.word_count[word.lemma_] for word in self.spacy_sent][:5]
        while len(output) < 5:
            output += [0]
        return output
    
    def set_answer_ner(self):
        output = [word.ent_type_ for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]][:5]
        return self.take5(output)
    
    def set_answer_ner_iob(self):
        output = [word.ent_iob_ for word in self.spacy_sent[self.answer_index_start:self.answer_index_end+1]][:5]
        return self.take5(output)

    def set_answer_srl_label(self):
        if len(self.srl) > 0:
            srl_labels = []
            for rel in self.srl:
                for key in rel:
                    if self.ascii_answer in rel[key]:
                        srl_labels += [key]
            if len(srl_labels) > 0:
                return Counter(srl_labels).most_common(1)[0][0]
            else:
                return 'XX'
        else:
            return 'XX'
    
    def take5(self, output):
        output = output[:5]
        while len(output) < 5:
            output.append('XX')
        return output
    
    def vector(self):
        vector = []
        vector += [self.answer_length, self.answer_depth]
        vector += self.answer_word_count
        for tag in self.answer_pos:
            vector += list(([tag] == np.array(self.all_pos_tags)).astype('int'))
        for tag in self.answer_ner_iob:
            vector += list(([tag] == np.array(['I', 'O', 'B'])).astype('int'))
        for tag in self.answer_ner:
            vector += list(([tag] == np.array(self.all_ner_tags)).astype('int'))
        vector += list(([tag] == np.array(self.all_srl_labels)).astype('int'))
        return np.array(vector)

class Blanker(object):
    def __init__(self, sent_with_srl, nlp):
        self.nlp = nlp
        self.spacy = self.nlp(sent_with_srl['sentence'])
        self.srl = sent_with_srl['srl']
        self.blanks = self.make_blanks()
        
    def make_blanks(self):
        #good_tags = [u'NNP', u'NN', u'DT NN', u'NNS', u'CD', u'DT JJ NN', u'JJ NNS', u'NNP NNP', u'DT NN NN', u'DT NNP NNP', u'JJ NN', u'DT NNP', u'NN NN', u'NN NNS', u'JJ', u'CD NNS', u'DT NNS', u'VBD', u'DT NNPS', u'NNP POS', u'VB', u'JJ NNP', u'DT NNP NNP NNP', u'DT JJ NNS', u'DT NNP NN', u'CD NNP', u'NNP NNP NNP', u'DT JJ JJ NN', u'VBG', u'NNP CC NNP', u'NNP NN', u'DT NNP IN NNP', u'JJ NN NNS', u'DT JJ NN NN', u'NNP CD , CD', u'NNP CD', u'DT NNP NNP NNP NNP', u'NNP NNS', u'FW', u'PRP$ NNS']
        good_tags = [u'CD', u'JJ', u'VB', u'VBG', u'FW']

        #iterate thru some list of patterns
        matcher = spacy.matcher.Matcher(self.nlp.vocab)

        for i, pattern in enumerate(good_tags):
            matcher.add_pattern("i", [{spacy.attrs.TAG: tag} for tag in pattern.split()])

        matches = matcher(self.spacy) # this has to be a spacy blanked_sentence, so we have to run nlp(sentence) to get this

        noun_ent_matches = []
        for chunk in self.spacy.noun_chunks:
            noun_ent_matches.append((0,0, chunk.start, chunk.end))
        for ent in self.spacy.ents:
            noun_ent_matches.append((0,0, ent.start, ent.end))
        for match in matches:
            noun_ent_matches.append(match)

        # now we need to generate the actual question sentences with blanks, this is just for one sentence
        all_blanks = []
        for m in noun_ent_matches:
            spacy_answer = self.spacy[m[2]:m[3]]
            answer = ""
            blanked_sentence = ""
            for i, token in enumerate(self.spacy):
                if i in xrange(m[2], m[3]):
                    answer += (token.text+token.whitespace_)
                    blanked_sentence += ('_'+token.whitespace_)
                else:
                    blanked_sentence += (token.text+token.whitespace_)
            all_blanks.append({'question': blanked_sentence, 'answer': answer, 'sentence': self.spacy.text, 'spacy_answer': spacy_answer, 'srl':self.srl})

        return all_blanks