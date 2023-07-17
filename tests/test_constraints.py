import unittest
from collie.constraints import (
    TargetLevel,
    InputLevel,
    Relation,
    Reduction,
    Count,
    Position,
    Max,
    ForEach,
    Constraint,
    And,
)


class TestCombinedConstraints(unittest.TestCase):
    def test_and_transformations(self):
        c = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=And(ForEach(...), ForEach(Position(-1))),
            relation=Relation('in'),
            reduction=Reduction('at least', 2),
        )
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This sentence is a slightly longer fifth line.', 'sentence')
        self.assertTrue(result)
    
    def test_and_constraints(self):
        c_1 = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=ForEach(...),
            relation=Relation('in'),
            reduction=Reduction('at least', 2),
        )
        c_2 = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=ForEach(Position(-1)),
            relation=Relation('in'),
            reduction=Reduction('at least', 2),
        )
        c = And(c_1, c_2)
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This sentence is a slightly longer fifth line.', 'sentence')
        self.assertTrue(result)


class TestCharLevelConstraints(unittest.TestCase):
    def test_num_chars(self):
        c = Constraint(
            target_level=TargetLevel('character'),
            transformation=Count(),
            relation=Relation('=='),
        )
        result = c.check('good', 4)
        self.assertTrue(result)
        result = c.check('good', 3)
        self.assertFalse(result)

    def test_word_contains_chars(self):
        c = Constraint(
            input_level=InputLevel('word'),
            target_level=TargetLevel('character'),
            transformation=ForEach(...),
            relation=Relation('in'),
            reduction=Reduction('any'),
        )
        result = c.check('good', 'g')
        self.assertTrue(result)
        result = c.check('good', 'k')
        self.assertFalse(result)


class TestWordLevelConstraints(unittest.TestCase):
    def test_num_words(self):
        c = Constraint(
            target_level=TargetLevel('word'), 
            transformation=Count(),
            relation=Relation('=='),
        )
        result = c.check('This is a good sentence.', 5)
        self.assertTrue(result)
        result = c.check('This is a good sentence.', 4)
        self.assertFalse(result)
    
    def test_word_position(self):
        c = Constraint(
            target_level=TargetLevel('word'), 
            transformation=Position(3),
            relation=Relation('=='),
        )
        result = c.check('This is a good sentence.', 'good')
        self.assertTrue(result)
        result = c.check('This is a good sentence.', 'is')
        self.assertFalse(result)
    
    def test_max_words_per_sentence(self):
        c = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=Max(ForEach(Count())),
            relation=Relation('<='),
        )
        result = c.check('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 7)
        self.assertTrue(result)
        result = c.check('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 5)
        self.assertFalse(result)
    
    def test_all_sentences_end_with_word(self):
        c = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=ForEach(Position(-1)),
            relation=Relation('=='),
            reduction=Reduction('all'),
        )
        result = c.check('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 'sentence')
        self.assertTrue(result)
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'sentence')
        self.assertFalse(result)
    
    def test_all_sentences_contain_word(self):
        c = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=ForEach(...),
            relation=Relation('in'),
            reduction=Reduction('all'),
        )
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'This')
        self.assertTrue(result)
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'chicken')
        self.assertFalse(result)
    
    def test_at_least_one_sentence_contains_word(self):
        c = Constraint(
            input_level=InputLevel('sentence'),
            target_level=TargetLevel('word'), 
            transformation=ForEach(...),
            relation=Relation('in'),
            reduction=Reduction('at least', 2),
        )
        result = c.check('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'sentence')
        self.assertTrue(result)
