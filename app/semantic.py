class Sentence(object):
    def __init__(self, sentence, question, answer):
        self.raw_sentence = sentence
        self.raw_question = question
        self.raw_answer = answer
        self.ascii_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
        self.ascii_question = unicodedata.normalize('NFKD', question).encode('ascii','ignore')
        self.ascii_answer = unicodedata.normalize('NFKD', answer).encode('ascii','ignore')
        self.spacy_sent = nlp(self.raw_sentence)
        self.spacy_ques = nlp(self.raw_question)
        self.answer_length = self.set_answer_length()
        self.spacy_answer = self.set_spacy_answer()
        self.srl = annotator.getAnnotations(self.ascii_sentence)['srl']
        self.answer_pos = self.set_answer_pos()
        self.answer_ner = self.set_answer_ner()
        self.answer_ner_iob  = self.set_answer_ner_iob()
        self.answer_srl_label = self.set_answer_srl_label()
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