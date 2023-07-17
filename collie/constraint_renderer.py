import numpy as np
from typing import Union, List, Any
from collie.constraints import *
import openai
import os

class ConstraintRenderer:
    # Mapping of relations to their respective text
    relation_map: dict = {
        '==': 'exactly',
        '>=': 'at least',
        '<=': 'at most',
        '<': 'less than',
        '>': 'more than',
        'in': 'in',
        'not in': 'not in'
    }

    levels: List[str] = ['character', 'word', 'sentence', 'paragraph', 'passage']
    # Mapping of levels to their respective numbers
    levels_dict: dict = {
        'character': 1,
        'word': 2,
        'sentence': 3,
        'paragraph': 4,
        'passage': 5
    }

    def __init__(self, constraint: Union[Constraint, Logic], check_value: Any, gpt_polish: bool = False, enforce_g_level: str = None):
        self.constraint = constraint
        # If constraint is an instance of Constraint, initialize necessary variables
        if isinstance(constraint, Constraint):
            self.t_level = constraint.target_level.level
            if constraint.input_level:
                self.g_level = constraint.input_level.level if constraint.input_level.level else self.upgrade_g_level(self.t_level)
            else:
                self.g_level = self.upgrade_g_level(self.t_level)
            if enforce_g_level:
                self.g_level = enforce_g_level
            self.relation = self.relation_map[constraint.relation.operand]
            self.check_value = check_value
            self.reduction = (constraint.reduction.reduction, constraint.reduction.value)
            self.prompt = self.render_prompts_single_constraint(constraint)
            if gpt_polish:
                self.polished_prompt = self.polish_prompt(self.prompt)
        # If constraint is an instance of Logic, parse the constraints and render the prompts
        elif isinstance(constraint, Logic):
            self.g_level = enforce_g_level
            constrains = self.parse_constraints(constraint)
            self.check_value = check_value
            self.prompt = self.render_prompts_multiple_constraints(constrains)
            if gpt_polish:
                self.polished_prompt = self.polish_prompt(self.prompt)

    def render_prompts_single_constraint(self, constraint: Constraint, check_value: Any = None, feedback_mode: bool = False) -> str:
        # Assign the check_value if provided
        check_value = check_value if check_value else self.check_value

        # Parse the transformation and create a flattened list
        flattend_transformation = self.parse_transform(constraint.transformation)

        # Draft the initial prompt
        prompt = self.draft_prompt(feedback_mode=feedback_mode)

        # Add the constraint to the drafted prompt
        prompt = self.add_constraint_to_prompt(prompt, flattend_transformation, check_value=check_value, feedback_mode=feedback_mode)

        return prompt

    def render_prompts_multiple_constraints(self, constraints: Union[List, Constraint], check_value: Any = None, feedback_mode: bool = False) -> str:
        # Assign the check_value if provided
        check_value = check_value if check_value else self.check_value

        # Handles nested constraints and logic operations
        # (e.g., ["and", [constraint1, constraint2]], ["all", [constraint1, constraint2, constraint3]])
        if isinstance(constraints, list):
            if constraints[0] in ["and", "or"]:
                prompt = [constraints[0], 
                            [self.render_prompts_multiple_constraints(constraints[1][0], check_value[0], feedback_mode),
                             self.render_prompts_multiple_constraints(constraints[1][1], check_value[1], feedback_mode)]]
                g_level_ind = np.argmax(self.level_dict[x.split(" ")[3]] for x in prompt[1])
                self.g_level = prompt[1][g_level_ind].split(" ")[3]
                if not feedback_mode:
                    synthesized = " ".join(prompt[1][g_level_ind].split(" ")[:5]) + " "
                    synthesized += " ".join(prompt[1][0].split(" ")[5:])[:-1] + f" {constraints[0]} " 
                    synthesized += " ".join(prompt[1][1].split(" ")[5:])
                else:
                    synthesized = " ".join(prompt[1][g_level_ind].split(" ")[:5]) + " "
                    synthesized += " ".join(prompt[1][0].split(" ")[5:])[:-1] + f" {constraints[0]} " 
                    synthesized += " ".join(prompt[1][1].split(" ")[5:])
                return synthesized
            elif constraints[0] in ["all"]:
                prompt = [constraints[0], 
                            [self.render_prompts_multiple_constraints(constraints[1][i], check_value[i], feedback_mode) 
                                for i in range(len(constraints[1]))]]
                g_level_ind = np.argmax(self.level_dict[x.split(" ")[3]] for x in prompt[1])
                if self.g_level is None:
                    self.g_level = prompt[1][g_level_ind].split(" ")[3]
                if not feedback_mode:
                    synthesized = " ".join(prompt[1][g_level_ind].split(" ")[:4]) + ":\n"
                    for i in range(len(prompt[1])):
                        if i != len(prompt[1]) - 1:
                            synthesized += f"{i+1}) " + " ".join(prompt[1][i].split(" ")[4:])[:-1] + f";\n" 
                        else:
                            synthesized += f"{i+1}) " + " ".join(prompt[1][i].split(" ")[4:]) 
                else:
                    synthesized = " ".join(prompt[1][g_level_ind].split(" ")[:5]) + ":\n"
                    for i in range(len(prompt[1])):
                        if i != len(prompt[1]) - 1:
                            synthesized += f"{i+1}) " + " ".join(prompt[1][i].split(" ")[5:])[:-1] + f";\n" 
                        else:
                            synthesized += f"{i+1}) " + " ".join(prompt[1][i].split(" ")[5:]) 

                return synthesized
        else:
            # If the constraints are a single Constraint object, render the prompt for the constraint
            if isinstance(constraints, Constraint):
                reduction = (constraints.reduction.reduction, constraints.reduction.value)
                relation = self.relation_map[constraints.relation.operand]
                t_level = constraints.target_level.level
                g_level = constraints.input_level.level if constraints.input_level.level else self.upgrade_g_level(t_level)
                
                prompt = self.draft_prompt(reduction=reduction, g_level=g_level, t_level=t_level, feedback_mode=feedback_mode)
                flattend_transformation = self.parse_transform(constraints.transformation)
                prompt = self.add_constraint_to_prompt(prompt, flattend_transformation,
                                                       relation=relation, g_level=g_level, 
                                                       t_level=t_level, check_value=check_value, feedback_mode=feedback_mode)
            return prompt
        return "To Be Implemented."

    def polish_prompt(self, prompt: str):
        openai.organization = os.environ.get("OPENAI_ORG")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        res = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant to polish sentence in natural language."
                        },
                        {
                            "role": "user",
                            "content": f"Please rewrite the following paragraph to be more fluent, without changing the original meaning. You should directly output the revised paragraph. The original paragraph:\n {prompt}"
                        }
                    ],
                    temperature = 0
                )
        msg = res.get("choices")[0]["message"]["content"]
        return msg

    def add_constraint_to_prompt(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        # Set the default values for g_level, t_level, relation, and check_value if not provided
        g_level = g_level if g_level else self.g_level
        t_level = t_level if t_level else self.t_level
        relation = relation if relation else self.relation
        check_value = check_value if check_value else self.check_value

        # Add the constraint to the prompt based on the transformation type
        if not trans_list:
            return prompt

        if isinstance(trans_list[0], str):
            trans_type = trans_list[0]
            if trans_type == 'count':
                prompt = self.handle_count_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
            elif trans_type.split()[0] == 'the':
                prompt = self.handle_position_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
            elif trans_type == 'each':
                prompt = self.handle_foreach_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
            elif trans_type in ['max', 'min']:
                prompt = self.handle_maxmin_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
            elif trans_type in ['and', 'or']:
                prompt = self.handle_logic_andor_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
            elif trans_type in ['all']:
                prompt = self.handle_logic_all_transformation(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)                

        return prompt

    def handle_count_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        if isinstance(check_value, list) and len(check_value) == 1:
            check_value = check_value[0]
        if len(trans_list) > 1:
            if not feedback_mode:
                prompt = prompt.replace("@...", relation + " " + str(check_value)) + f" \'{trans_list[1][0]}\'."
            else:
                if isinstance(check_value, list):
                    if relation in ['at least', 'more than']:
                        check_value = min(check_value)
                    elif relation in ['at most', 'less than']:
                        check_value = max(check_value)
                    elif relation in ['exactly']:
                        check_value = f"{handle_value(check_value)}"
                prompt = prompt.replace("@...", str(check_value)) + f" \'{trans_list[1][0]}\'."
            trans_list = trans_list[2:]
        else:
            if not feedback_mode:
                prompt = prompt.replace("@...", relation + " " + str(check_value)) + ("s." if check_value > 1 else ".")
            else:
                if isinstance(check_value, list):
                    if relation in ['at least', 'more than']:
                        check_value = min(check_value)
                    elif relation in ['at most', 'less than']:
                        check_value = max(check_value)
                    elif relation in ['exactly']:
                        check_value = f"{handle_value(check_value)}"
                prompt = prompt.replace("@...", str(check_value)) + ("." if check_value != 1 else "s.")
            trans_list = trans_list[1:]
        if t_level in ['character'] and g_level in ['sentence', 'paragraph', 'passage']:
            if not feedback_mode:
                prompt = prompt + " Include whitespace into your character count."
        return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    def handle_position_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        if not feedback_mode:
            if len(trans_list[0].split()) > 2:
                prompt = prompt.replace("@...", trans_list[0]) + f"s to be {self.handle_value(check_value)} respectively."
            else:
                if isinstance(check_value, list) and len(check_value) > 1:
                    prompt = prompt.replace("@...", trans_list[0]).replace('all ', '') + f" to be {self.handle_value(check_value)}, respectively."
                else:
                    prompt = prompt.replace("@...", trans_list[0]) + f" to be {self.handle_value(check_value)}."
        else:
            if len(trans_list[0].split()) > 2:
                prompt = prompt.replace("@...", trans_list[0]) + f"s being {self.handle_value(check_value)} respectively."
            else:
                if isinstance(check_value, list) and len(check_value) > 1:
                    prompt = prompt.replace("@...", trans_list[0]).replace('all ', '') + f" to be {self.handle_value(check_value)}, respectively."
                else:
                    prompt = prompt.replace("@...", trans_list[0]) + f" being {self.handle_value(check_value)}."
        trans_list = trans_list[1:]
        return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    def handle_foreach_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        if trans_list[1] == [] and relation == "in":
            if "with @..." in prompt:
                prompt = prompt.replace("with @...", "containing the") + f" {self.handle_value(check_value)}."
            else:
                prompt = prompt.replace("@...", "containing the") + f" {self.handle_value(check_value)}."
            trans_list = trans_list[2:]
            return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
        elif trans_list[1] == [] and relation == "not in":
            if not feedback_mode:
                if "with @..." in prompt:
                    prompt = prompt.replace("with @...", "not containing the") + f" {self.handle_value(check_value)}."
                else:
                    prompt = prompt.replace("@...", "not containing the") + f" {self.handle_value(check_value)}."
            else:
                if "with @..." in prompt:
                    prompt = prompt.replace("with @...", "containing the") + f" {self.handle_value(check_value)}."
                else:
                    prompt = prompt.replace("@...", "containing the") + f" {self.handle_value(check_value)}."
            trans_list = trans_list[2:]
            return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)
        elif isinstance(trans_list[1], list) and len(trans_list[1]) > 0:
            if "with @..." in prompt:
                prompt = prompt.replace("with @...", "having @...")
            else:
                prompt = prompt.replace("@...", "having @...")
            trans_list = trans_list[1]
            return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    def handle_maxmin_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        prompt = prompt.replace(g_level + " with", f"{self.upgrade_g_level(g_level)} with each {g_level}")
        trans_list = trans_list[1]
        return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    def handle_logic_andor_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        trans_list_sub1 = trans_list[1][0]
        trans_list_sub2 = trans_list[1][1]
        prompt = self.add_constraint_to_prompt(prompt, trans_list_sub1, g_level, t_level, relation, check_value, feedback_mode)
        prompt = prompt[:-1] + f" {trans_list[0]} @... {t_level}"
        prompt = self.add_constraint_to_prompt(prompt, trans_list_sub2, g_level, t_level, relation, check_value, feedback_mode)
        trans_list = trans_list[2:]
        return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    def handle_logic_all_transformation(self, prompt: str, trans_list: List, g_level: str = None, t_level: str = None, relation: str = None, check_value: Any = None, feedback_mode: bool = False) -> str:
        trans_list_subs = trans_list[1]
        for i, trans_sub in enumerate(trans_list_subs):
            if i == 0: 
                prompt = prompt.replace("@...", "(1) @...")
            prompt = self.add_constraint_to_prompt(prompt, trans_sub, g_level, t_level, relation, check_value, feedback_mode)
            prompt = prompt[:-1] + f" and ({i+1}) @... {t_level}"
        trans_list = trans_list[2:]
        return self.add_constraint_to_prompt(prompt, trans_list, g_level, t_level, relation, check_value, feedback_mode)

    
    def handle_value(self, check_value: Any = None) -> str:
        if isinstance(check_value, list):
            res = ""
            for item in check_value:
                if isinstance(item, str):
                    if res == "":
                        res = f"\'{item}\'"
                    else:
                        res = res + f", \'{item}\'"
                elif isinstance(item, int):
                    if res == "":
                        res = f"{item}"
                    else:
                        res = res + f", {item}"
                else:
                    if res == "":
                        res = f"\'(empty)\'"
                    else:
                        res = res + f", \'(empty)\'"
            return res
        elif isinstance(check_value, str):
            return f"\'{check_value}\'"
        elif isinstance(check_value, int):
            return f"{check_value}"
        else:
            return "\'(empty)\'"


    def parse_transform(self, trans: Union[Position, Count, Logic]) -> list:
        """
        Parses the transformation depending on its type and returns the result in a specific format.
        """
        if isinstance(trans, Position):
            if isinstance(trans.position, int):
                return [f"the {self.ordinal(trans.position + 1)}"]
            elif isinstance(trans.position, list):
                return [f"the {', '.join([self.ordinal(p + 1) for p in trans.position])}"]

        elif isinstance(trans, Count):
            if trans.count_target:
                return ["count", [trans.count_target]]
            else:
                return ["count"]

        elif isinstance(trans, Logic):
            if isinstance(trans, And):
                return ["and", [self.parse_transform(trans.callable_1), self.parse_transform(trans.callable_2)]]
            elif isinstance(trans, Or):
                return ["or", [self.parse_transform(trans.callable_1), self.parse_transform(trans.callable_2)]]
            elif isinstance(trans, All):
                res = ["all"]
                all_list = []
                for k, callable_fn in enumerate(trans.callables):
                    all_list.append(f"{k+1}) {self.parse_transform(callable_fn)}")
                res.append(all_list)
                return res

        elif hasattr(trans, "func"):
            if isinstance(trans, ForEach):
                return ["each", self.parse_transform(trans.func)]
            elif isinstance(trans, Max):
                return ["max", self.parse_transform(trans.func)]
            else:
                return [self.parse_transform(trans.func)]

        return []

    def parse_constraints(self, constraint: Union[Logic, Constraint]) -> Union[list, Constraint]:
        """
        Parses the constraints depending on its type and returns the result in a specific format.
        """
        if isinstance(constraint, Logic):
            if isinstance(constraint, And):
                return ["and", [self.parse_constraints(constraint.callable_1), 
                                self.parse_constraints(constraint.callable_2)]]
            elif isinstance(constraint, Or):
                return ["or", [self.parse_constraints(constraint.callable_1), 
                                self.parse_constraints(constraint.callable_2)]]
            elif isinstance(constraint, All):
                return ["all", [self.parse_constraints(callable_x) 
                                for callable_x in constraint.callables]]
        elif isinstance(constraint, Constraint):
            return constraint

    def draft_prompt(self, reduction: tuple = None, g_level: str = None, t_level: str = None, feedback_mode: bool = False) -> str:
        """
        Drafts the initial prompt based on the given reduction, g_level, and t_level.
        """
        reduction = reduction if reduction else self.reduction
        g_level = g_level if g_level else self.g_level
        t_level = t_level if t_level else self.t_level
        generate_with_tense = "just generated" if feedback_mode else "generate"
        if reduction[0] in ['at least', 'at most', 'exactly']:
            if not feedback_mode:
                if reduction[1] > 1:
                    return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with {reduction[0]} {reduction[1]} {g_level}s @... {t_level}"
                else:
                    return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with {reduction[0]} {reduction[1]} {g_level} @... {t_level}"
            else:
                if reduction[1] > 1:
                    return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with {reduction[1]} {g_level}s @... {t_level}"
                else:
                    return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with {reduction[1]} {g_level} @... {t_level}"
        elif reduction[0] in ['all']:
            if not feedback_mode:
                return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with {reduction[0]} {g_level}s @... {t_level}"
            else:
                return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with some {g_level}s @... {t_level}"
        elif reduction[0] in ['any']:
            if not feedback_mode:
                return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with at least one {g_level}  @... {t_level}"
            else:
                return f"Please {generate_with_tense} a {self.upgrade_g_level(g_level)} with no {g_level}  @... {t_level}"
        else:
            return f"Please {generate_with_tense} a {g_level} with @... {t_level}" 


    def get_feedback(self, text: str, gpt_polish: bool = False) -> str:
        # If constraint is an instance of Constraint, initialize necessary variables
        if self.constraint.check(text, self.check_value):
            self.feedback = f"The generated {self.g_level} satisfies all constraints."
            return self.feedback
        else:
            observed_value = self.constraint.extract(text)
            if isinstance(self.constraint, Constraint):
                self.feedback = self.render_prompts_single_constraint(self.constraint, observed_value, feedback_mode=True)
                self.feedback = f"Your task is to {self.prompt[7:]}"[:-1]+f".\nHowever, you {self.feedback[7:]}"
                if gpt_polish:
                    self.feedback = self.polish_prompt(self.feedback)
            # If constraint is an instance of Logic, parse the constraints and render the prompts
            elif isinstance(self.constraint, Logic):
                constrains = self.parse_constraints(self.constraint)
                self.feedback = self.render_prompts_multiple_constraints(constrains, observed_value, feedback_mode=True)
                self.feedback = f"Your task is to {self.prompt[7:]}"[:-1]+f".\nHowever, you {self.feedback[7:]}"
                if gpt_polish:
                    self.feedback = self.polish_prompt(self.feedback)
            return self.feedback 


    def upgrade_g_level(self, g_level: str) -> str:
        """
        Upgrades the g_level by one level.
        """
        if self.levels_dict[g_level] < len(self.levels):
            return self.levels[self.levels_dict[g_level]]
        else:
            return "passage"

    def ordinal(self, n: int) -> str:
        """
        Converts an integer into its ordinal representation.
        """
        if n == -1 or n == 0:
            return "last"
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}
        return str(n) + suffix.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')
