import unittest
import nltk
import numpy as np
from app import helpers
import tests.data as test_data


class HelpersTest(unittest.TestCase):


	sentence = 'In Dahlem , there are several museums of world art and culture, such as the Museum of Asian Art , the Ethnological Museum , the Museum of European Cultures , as well as the Allied Museum (a museum of the Cold War) and the Brücke Museum (an art museum).'
	tagged_sentence = helpers.tag_sentence(sentence)
	question = 'In Dahlem , there are several museums of world art and culture, such as _ _ _ _ _  of Asian Art , the Ethnological Museum , the Museum of European Cultures , as well as the Allied Museum (a museum of the Cold War) and the Brücke Museum (an art museum).'
	tagged_question = helpers.tag_sentence(question)
	answer = 'the Museum'

	metadata = {'answer_depth': 0.2692307692307692,
 				'answer_length': 2,
 				'answer_pos': ['DT', 'NNP'],
 				'context_pos': ['JJ', 'IN', 'IN', 'NNP'],
 				'word_count': [6, 5]}
	
	vector = test_data.vector
	answer_pos_vector = test_data.answer_pos_vector
	word_count_vector = np.array([6, 5, 0, 0, 0])
	context_pos_vector = test_data.context_pos_vector

	def test_get_answer_pos_returns_proper_tags(self):
		self.assertEqual(['DT', 'NNP'], helpers.get_answer_pos(self.tagged_question, self.tagged_sentence, self.answer))

	def test_get_answer_depth(self):
		self.assertEqual(14/52, helpers.get_answer_depth(self.question))

	def test_get_context_pos_returns_proper_tags(self):
		self.assertEqual(['JJ', 'IN', 'IN', 'NNP'], helpers.get_context_pos(self.tagged_question, self.tagged_sentence, self.answer))

	def test_get_answer_word_count_returns_correct_word_count(self):
		self.assertEqual([6, 5], helpers.get_answer_word_count(self.sentence, self.answer))

	def test_get_metadata_returns_proper_metadata(self):
		self.assertEqual(self.metadata, helpers.get_metadata(self.sentence, self.tagged_sentence, self.question, self.tagged_question, self.answer))

	def test_sentence_vectorizer_returns_correct_vector(self):
		self.assertTrue(np.array_equal(self.vector, helpers.sentence_vectorizer(self.metadata)))

	def test_answer_pos_vectorizer_returns_correct_vector(self):
		self.assertTrue(np.array_equal(self.answer_pos_vector, helpers.answer_pos_vectorizer(self.metadata)))

	def test_word_count_vectorizer_returns_proper_vector(self):
		self.assertTrue(np.array_equal(self.word_count_vector, helpers.word_count_vectorizer(self.metadata)))

	def test_context_pos_vectorizer(self):
		self.assertTrue(np.array_equal(self.context_pos_vector, helpers.context_pos_vectorizer(self.metadata)))

	def test_get_trees_returns_correct_number_of_trees(self):
		trees = helpers.get_trees(self.sentence, helpers.get_top_patterns(3))[1]
		self.assertEqual(3, len(trees))

	def test_get_trees_returns_list_of_nltk_trees(self):
		trees = helpers.get_trees(self.sentence, helpers.get_top_patterns(3))[1]
		is_a_tree = [True if type(item) == nltk.tree.Tree else False for item in trees]
		self.assertTrue(all(is_a_tree))

	def test_get_trees_returns_tagged_sentence(self):
		tagged_sentence = helpers.get_trees(self.sentence, helpers.get_top_patterns(3))[0]
		self.assertEqual(self.tagged_sentence, tagged_sentence)

	def test_blankify_generates_proper_object_for_one_question(self):
		tree = helpers.get_trees(self.sentence, helpers.get_top_patterns(3))[1][0]
		self.assertEqual(helpers.blankify(tree)[1], test_data.sample_question)

	def test_make_all_questions_generates_list_of_dicts(self):
		output = helpers.make_all_questions(self.sentence, helpers.get_top_patterns(3))
		is_a_dict = [True if type(item) == dict else False for item in output]
		self.assertTrue(all(is_a_dict))