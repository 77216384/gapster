from nltk.tag import StanfordPOSTagger, StanfordNERTagger
from nltk import word_tokenize, pos_tag
from multiprocessing import Pool
import pandas as pd
import csv


def get_pos(sentence):
	index, text = sentence
	wt = word_tokenize(text)

	try:
		pos = st.tag(wt)
	except:
		print('using standard tagger')
		pos = pos_tag(wt)
	return (index, pos)


def concurrent_pos(recs, num_cores=2):
	pool = Pool(processes=num_cores)
	return pool.map(get_pos, recs)

jar = '/Users/alex/stanford-postagger-2017-06-09/stanford-postagger.jar'
model = '/Users/alex/stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
st = StanfordPOSTagger(model, jar)

df = pd.read_csv('mind_the_gap_dropna.csv', index_col=[0])
sentences = [(idx, text) for idx, text in enumerate(df['Sentence'])]
questions = [(idx, text) for idx, text in enumerate(df['Question'])]

sentence_pos = concurrent_pos(sentences[0:2])

try:
	with open("mind_the_gap_sent_pos.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(sentence_pos)
except:
	pd.DataFrame(sentence_pos).to_csv("mind_the_gap_sent_pos_df.csv")


question_pos = concurrent_pos(questions)

try:
	with open("mind_the_gap_ques_pos.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(question_pos[0:2])
except:
	pd.DataFrame(question_pos).to_csv("mind_the_gap__ques_pos_df.csv")