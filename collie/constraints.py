from typing import List, Any, Callable, Iterable
from nltk import word_tokenize, sent_tokenize
from rich import print
import string


def sus_target(t:str):
    # returns True if the target t is suspicious
    if not isinstance(t, str):
        return False 
    t = t.strip() # leading / trailing whitespace ok
    if t[0] not in string.ascii_letters + string.digits:
        return True # first letter not a letter
    if t[-1] not in string.ascii_letters + string.digits + string.punctuation:
        return True # last letter not a letter or punctuation.
    return False


class Level:
    _para_delim = "\n\n"

    def __call__(self, text):
        if self.level is None:
            return text
        if isinstance(text, str):
            tokenized = None
            if self.level == 'character':
                tokenized = list(text)
            elif self.level == 'word':
                tokenized = [x for x in word_tokenize(text) if x not in string.punctuation] 
            elif self.level == 'phrase':
                raise NotImplementedError
            elif self.level == 'sentence':
                tokenized = sent_tokenize(text)
            elif self.level == 'paragraph':
                tokenized = self.split_paragraphs(text)
            elif self.level == 'passage':
                raise NotImplementedError
            tokenized = [tok.strip().strip('.') for tok in tokenized]  # TODO: make this more general
            return tokenized
        elif isinstance(text, list):
            return [self(unit) for unit in text]
        else:
            raise ValueError(f'Input text must be a string or a list of strings, not {type(text)}.')
        
    @staticmethod
    def join_paragraphs(text:Iterable[str]) -> str:
        return Level._para_delim.join(text)

    @staticmethod
    def split_paragraphs(text:str) -> List[str]:
        return text.split(Level._para_delim) 



class InputLevel(Level):
    def __init__(
        self,
        level: str = None,
    ):
        super().__init__()
        self.level = level
    
    def __str__(self):
        return f'InputLevel({self.level})'


class TargetLevel(Level):
    def __init__(
        self,
        level: str = None,
    ):
        super().__init__()
        self.level = level
    
    def __str__(self):
        return f'TargetLevel({self.level})'


class Transformation:
    pass


class Count(Transformation):
    def __init__(self, count_target = None):
        super().__init__()
        self.count_target = count_target
    
    def __call__(self, units):
        if self.count_target is None:
            count = len(units)
        else:
            count = len([unit for unit in units if unit == self.count_target])
        return count
    
    def __str__(self):
        if self.count_target is None:
            return f'Count()'
        else:
            return f'Count({self.count_target})'


class Position(Transformation):
    def __init__(self, position=None):
        super().__init__()
        self.position = position
    
    @staticmethod
    def get(units, position):
        if len(units) == 0: return None
        if position >= len(units): return None
        if position < -len(units): return None
        return units[position]

    def __call__(self, units):
        if isinstance(self.position, int):
            return self.get(units, self.position)
        elif isinstance(self.position, list):
            return [self.get(units, i) for i in self.position]
        else:
            raise ValueError(f'Position must be an integer or a list of integers, not {type(self.position)}.')
    
    def __str__(self):
        return f'Position({self.position})'


class PositionOf(Transformation):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        units = self.func(x)
        return [i for i, u in enumerate(units)]
    
    def __str__(self):
        return f'PositionsOf({self.func})'


class MapPosition(Transformation):
    def __init__(self, func, map_pos_func):
        super().__init__()
        self.func = func
        self.map_pos_func = map_pos_func
    
    def __call__(self, x):
        units = self.func(x)
        return self.map_pos_func(units)
    
    def __str__(self):
        return f'MapPosition({self.func}, {self.map_func})'


class Aggregate(Transformation):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        # TODO: need to merge with the level class
        units = self.func(x)
        return ''.join(units)

        assert isinstance(units, list)
        if self.level == 'character':
            return ''.join(units)
        elif self.level == 'word':
            return ' '.join(units)
        elif self.level == 'phrase':
            raise NotImplementedError
        elif self.level == 'sentence':
            return '. '.join(units).strip()
        elif self.level == 'paragraph':
            raise NotImplementedError
        elif self.level == 'passage':
            raise NotImplementedError
        else:
            raise ValueError(f'Level {self.level} is not supported.')
    
    def __str__(self):
        return f'Aggregate({self.func})'


class Max(Transformation):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        units = self.func(x)
        return max(units)
    
    def __str__(self):
        return f'Max({self.func})'


class ForEach(Transformation):
    def __init__(self, func):
        self.func = func
    
    def __call__(self, units):
        if self.func is Ellipsis:
            func = lambda x: x
        else:
            func = self.func
        return [func(unit) for unit in units]
    
    def __str__(self):
        if self.func is Ellipsis:
            return f'ForEach(...)'
        else:
            return f'ForEach({self.func})'


class Logic:
    def check(self, x, target):
        return self(x, target)


class And(Logic):
    def __init__(self, callable_1, callable_2):
        super().__init__()
        self.callable_1 = callable_1
        self.callable_2 = callable_2
    
    def __call__(self, x, target=None):
        if target is None:
            return self.callable_1(x) and self.callable_2(x)
        elif isinstance(target, list):
            assert len(target) == 2
            return self.callable_1(x, target[0]) and self.callable_2(x, target[1])
        else:
            return self.callable_1(x, target) and self.callable_2(x, target)
    
    def __str__(self):
        return f'And({self.callable_1}, {self.callable_2})'
    
    def extract(self, x):
        return [callable_.extract(x) for callable_ in self.callables]


class Or(Logic):
    def __init__(self, callable_1, callable_2):
        super().__init__()
        self.callable_1 = callable_1
        self.callable_2 = callable_2
    
    def __call__(self, x, target=None):
        if target is None:
            return self.callable_1(x) or self.callable_2(x)
        elif isinstance(target, list):
            assert len(target) == 2
            return self.callable_1(x, target[0]) or self.callable_2(x, target[1])
        else:
            return self.callable_1(x, target) or self.callable_2(x, target)
    
    def __str__(self):
        return f'Or({self.callable_1}, {self.callable_2})'
    
    def extract(self, x):
        return [callable_.extract(x) for callable_ in self.callables]


class All(Logic):
    def __init__(self, *callables):
        super().__init__()
        self.callables = callables
    
    def __call__(self, x, target=None):
        if target is None:
            return all([callable_(x) for callable_ in self.callables])
        elif isinstance(target, list):
            return all([callable_(x, t) for callable_, t in zip(self.callables, target)])
        else:
            raise NotImplementedError
    
    def __str__(self):
        return f"All({', '.join([str(c) for c in self.callables])})"
    
    def extract(self, x):
        return [callable_.extract(x) for callable_ in self.callables]


class Relation:
    """
    Abstract relation class that works for more literal types.
    """
    def __init__(self, operand):
        self.operand = operand

    def _patch_literal(self, literal:str):
        # apply transformations on literal to remove artifacts and casing

        def _patch(literal:str):
            # apply the patch if it is 
            if not isinstance(literal, str):
                return literal
            stripped = literal.lower().strip(string.punctuation + " ")
            return literal if stripped == "" else stripped
        
        if isinstance(literal, list):
            return [_patch(x) for x in literal]
        
        return _patch(literal)

    def __call__(self, literal_1, literal_2):
        literal_1, literal_2 = self._patch_literal(literal_1), self._patch_literal(literal_2) 
        if self.operand in ["==", "!=", "<", ">", "<=", ">="]:
            if isinstance(literal_2, list) and len(literal_2) == 1:
                literal_2 = literal_2[0]
            if self.operand == '==':
                return literal_1 == literal_2
            elif self.operand == '!=':
                return literal_1 != literal_2
            elif self.operand == '<':
                return literal_1 < literal_2
            elif self.operand == '<=':
                return literal_1 <= literal_2
            elif self.operand == '>':
                return literal_1 > literal_2
            elif self.operand == '>=':
                return literal_1 >= literal_2
        elif self.operand in ["in", "not in"]:
            if self.operand == 'in':
                if isinstance(literal_2, list):
                    return all([l in literal_1 for l in literal_2])
                else:
                    return literal_2 in literal_1
            elif self.operand == 'not in':
                if isinstance(literal_2, list):
                    return not any([l in literal_1 for l in literal_2])
                else:
                    return literal_2 not in literal_1
        else:
            raise NotImplementedError
    
    def __str__(self):
        return f'Relation({self.operand})'


class Reduction:
    def __init__(self, reduction=None, value=None):
        """
        reduction (str): one of ['at least', 'at most', 'exactly', 'all', 'any']
        """
        self.reduction = reduction
        self.value = value
    
    def __call__(self, x, target, relation):
        if self.reduction is None:
            return relation(x, target)
        if not isinstance(target, list):
            target = [target] * len(x)
        if len(x) != len(target): return False
        # assert len(x) == len(target), f'Length of x ({len(x)}) and target ({len(target)}) must be the same.'
        results = [relation(x_i, target_i) for x_i, target_i in zip(x, target)]

        if self.reduction == 'all':
            return all(results)
        elif self.reduction == 'any':
            return any(results)
        elif self.reduction == 'at least':
            return sum(results) >= self.value
        elif self.reduction == 'at most':
            return sum(results) <= self.value
        elif self.reduction == 'exactly':
            return sum(results) == self.value

    def __str__(self):
        if self.value is not None:
            return f'Reduction({self.reduction} {self.value})'
        else:
            return f'Reduction({self.reduction})'


class Constraint:
    def __init__(
        self,
        input_level: InputLevel = None,
        target_level: TargetLevel = None,
        transformation: Callable = None,
        relation: Relation = None,
        reduction = None,
    ):
        self.input_level = input_level or InputLevel()
        self.target_level = target_level or TargetLevel()
        self.transformation = transformation
        self.relation = relation
        self.reduction = reduction or Reduction()
    
    def extract(self, text):
        if self.input_level is not None:
            input_units = self.input_level(text)
        else:
            input_units = text
        x = self.target_level(input_units)
        x = self.transformation(x)
        return x
    
    def check(self, text, target):
        x = self.extract(text)
        return self.reduction(x, target, self.relation)
    
    def __call__(self, text, target):
        return self.check(text, target)
    
    def __str__(self):
        return (
            f'Constraint(\n'
            f'    {self.input_level},\n'
            f'    {self.target_level},\n'
            f'    Transformation({self.transformation}),\n'
            f'    {self.relation},\n'
            f'    {self.reduction}\n'
            f')'
        )
    def __repr__(self):
        return self.__str__()