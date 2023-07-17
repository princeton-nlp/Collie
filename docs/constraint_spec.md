# Constraint Specification Guide

Below is an example of specifying a "counting the number of word constraint".
```python
from collie.constraints import Constraint, TargetLevel, Count, Relation

# A very simple "number of word" constraint.
>>> c = Constraint(
>>>     target_level=TargetLevel('word'),
>>>     transformation=Count(), 
>>>     relation=Relation('=='),
>>> )
>>> print(c)
Constraint(
    InputLevel(None),
    TargetLevel(word),
    Transformation('Count()'),
    Relation(==),
    Reduction(None)
)
>>> text = 'This is a good sentence.'
>>> print(c.check(text, 5))
True
>>> print(c.check(text, 4))
False
```

A slightly more complex example "the max number of words in each sentence is less and equal to" constraint.
```python
from collie.constraints import Constraint, InputLevel, TargetLevel, Max, Count, ForEach, Relation

>>> c = Constraint(
>>>     # Look at input text at the sentence level
>>>     input_level=InputLevel('sentence'),
>>>     # Look at word when comparing to the target
>>>     target_level=TargetLevel('word'),
>>>     # Count number of words (TargetLevel) for each sentence (InputLevel) and take the max of these
>>>     transformation=Max(ForEach(Count())),
>>>     relation=Relation('<='),
>>> )
>>> print(c)
Constraint(
    InputLevel(sentence),
    TargetLevel(word),
    Transformation(Max(ForEach(Count()))),
    Relation(<=),
    Reduction(None)
)
>>> text = 'This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.'
>>> print(c.check(text, 7))
True
>>> print(c.check(text, 5))
False
```

More examples below.
```python
from rich import print
from collie.constraints import *


def print_results(constraint, text, check_value, result):
    print(constraint)
    print(f'text = "{text}"')
    check_value =  '"' + check_value + '"' if isinstance(check_value, str) else check_value
    print(f'check_value = {check_value}')
    print(f'result = {result}')
    print('----------------------')


if __name__ == '__main__':
    # NumWordsConstraint
    c1 = Constraint(
        target_level=TargetLevel('word'), 
        transformation=Count(),
        relation=Relation('=='),
    )
    p1 = [
        ('This is a good sentence.', 5),
        ('This is a good sentence.', 4),
    ]
    for text, check_value in p1:
        result = c1.check(text, check_value)
        print_results(c1, text, check_value, result)
    
    # WordPositionConstraint
    c2 = Constraint(
        target_level=TargetLevel('word'), 
        transformation=Position(3),
        relation=Relation('=='),
    )
    p2 = [
        ('This is a good sentence.', 'good'),
        ('This is a good sentence.', 'is'),
    ]
    for text, check_value in p2:
        result = c2.check(text, check_value)
        print_results(c2, text, check_value, result)
    
    # MaxNumberOfWordsPerSentenceConstraint
    c3 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=Max(ForEach(Count())),
        relation=Relation('<='),
    )
    # NOTE: there's another way to do this, which doesn't require the Max().
    # This can be done with Reduction('at most'), but this will require 
    # knowing the max number before hand.
    p3 = [
        ('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 7),
        ('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 5),
    ]
    for text, check_value in p3:
        result = c3.check(text, check_value)
        print_results(c3, text, check_value, result)


    # Notin
    c13 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(...),
        relation=Relation('not in'),
        reduction=Reduction('all')
    )
    p13 = [
        ('This is a sentence. This is another sentence.', 'sentence'),
        ('This is a sentence. This is another sentence.', 'a'),
    ]
    for text, check_value in p13:
        result = c13.check(text, check_value)
        print_results(c13, text, check_value, result)
    
    # AllSentenceEndsWithWordConstraint
    c4 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(Position(-1)),
        relation=Relation('=='),
        reduction=Reduction('all'),
    )
    p4 = [
        ('This is a sentence. This is another sentence. This is the third sentence. This is the fourth sentence. This is a slightly longer fifth sentence.', 'sentence'),
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'sentence'),
    ]
    for text, check_value in p4:
        result = c4.check(text, check_value)
        print_results(c4, text, check_value, result)
    
    # AllSentenceContainsWithWordConstraint
    c5 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(...),
        relation=Relation('in'),
        reduction=Reduction('all'),
    )
    p5 = [
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'This'),
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'chicken'),
    ]
    for text, check_value in p5:
        result = c5.check(text, check_value)
        print_results(c5, text, check_value, result)
    
    # AtLeastOneSentenceContainsWordConstraint
    c6 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(...),
        relation=Relation('in'),
        reduction=Reduction('at least', 2),
    )
    p6 = [
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This is a slightly longer fifth string.', 'sentence'),
    ]
    for text, check_value in p6:
        result = c6.check(text, check_value)
        print_results(c6, text, check_value, result)


    # AtLeastOneSentenceContainsWordConstraint
    c7 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=And(ForEach(...), ForEach(Position(-1))),
        relation=Relation('in'),
        reduction=Reduction('at least', 2),
    )
    p7 = [
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This sentence is a slightly longer fifth line.', 'sentence'),
    ]
    for text, check_value in p7:
        result = c7.check(text, check_value)
        print_results(c7, text, check_value, result)


    # AtLeastOneSentenceContainsWordConstraint
    c8_1 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(...),
        relation=Relation('in'),
        reduction=Reduction('at least', 2),
    )
    c8_2 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'), 
        transformation=ForEach(Position(-1)),
        relation=Relation('in'),
        reduction=Reduction('at least', 2),
    )
    c8 = And(c8_1, c8_2)
    p8 = [
        ('This is a sentence. This is another sentence. This is the third utterance. This is the fourth line. This sentence is a slightly longer fifth line.', 'sentence'),
    ]
    for text, check_value in p8:
        result = c8.check(text, check_value)
        print_results(c8, text, check_value, result)

    # WordContainsNCharsAndMCharsConstraint
    c10_1 = Constraint(
        target_level=TargetLevel('character'), 
        transformation=Count('v'),
        relation=Relation('=='),
    )
    c10_2 = Constraint(
        target_level=TargetLevel('character'), 
        transformation=Count('i'),
        relation=Relation('=='),
    )
    c10 = And(c10_1, c10_2)
    p10 = [
        ('vivacious', [2, 2]),
        ('vivacious', [2, 3]),
    ]
    for text, check_value in p10:
        result = c10.check(text, check_value)
        print_results(c10, text, check_value, result)

    # Generate a sentence with 26 words, where the 1st, 2th, and 26th words are “A”, “B”, “Z", respectively.
    c11_1 = Constraint(
        target_level=TargetLevel('word'),
        transformation=Position([0, 1, 25]),
        relation=Relation('=='),
        reduction=Reduction('all'),
    )
    c11_2 = Constraint(
        target_level=TargetLevel('word'),
        transformation=Count(),
        relation=Relation('=='),
    )
    c11 = And(c11_1, c11_2)
    p11 = [
        ('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z', [['A', 'B', 'Z'], 25]),
        ('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z', [['A', 'B', 'Y'], 25]),
        ('A B C D E F G H I J K L M N O P Q R S T U V W X Y Z', [['A', 'B', 'Z'], 26]),
    ]
    for text, check_value in p11:
        result = c11.check(text, check_value)
        print_results(c11, text, check_value, result)

    # Generate a paragraph with 3 sentences, 
    # each sentence with 5 words exactly, 
    # with exactly 4 "A" and 5 "B" in the paragraph.
    c12_1 = Constraint(
        target_level=TargetLevel('sentence'),
        transformation=Count(),
        relation=Relation('=='),
    )
    c12_2 = Constraint(
        input_level=InputLevel('sentence'),
        target_level=TargetLevel('word'),
        transformation=ForEach(Count()),
        relation=Relation('=='),
        reduction=Reduction('all'),
    )
    c12_3 = Constraint(
        target_level=TargetLevel('word'),
        transformation=Count('A'),
        relation=Relation('=='),
    )
    c12_4 = Constraint(
        target_level=TargetLevel('word'),
        transformation=Count('B'),
        relation=Relation('=='),
    )
    c12 = All(c12_1, c12_2, c12_3, c12_4)
    p12 = [
        ('A B C D E. A B C D E. A A B B B.', [3, 5, 4, 5]),
    ]
    for text, check_value in p12:
        result = c12.check(text, check_value)
        print_results(c12, text, check_value, result)
```