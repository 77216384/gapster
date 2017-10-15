from nltk import word_tokenize, pos_tag
from multiprocessing import Pool
import pandas as pd
import csv


def get_pos(sentence):
	index, text = sentence
	print(text[0:20])
	wt = word_tokenize(text)
	pos = pos_tag(wt)
	return (index, pos)


def concurrent_pos(sentences, num_cores=2):
	pool = Pool(processes=num_cores)
	return pool.map(get_pos, sentences)

df = pd.read_csv('mind_the_gap_dropna.csv', index_col=[0])
sentences = [(idx, text) for idx, text in enumerate(df['Sentence'])]
questions = [(idx, text) for idx, text in enumerate(df['Question'])]

sentence_pos = concurrent_pos(sentences)

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
		writer.writerows(question_pos)
except:
	pd.DataFrame(question_pos).to_csv("mind_the_gap__ques_pos_df.csv")