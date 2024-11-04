import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import random

class State:
    def __init__(self):
        self.assignments: Dict[str, List[str]] = defaultdict(list)  # desafio -> [estudiantes]
        self.unassigned_students: Set[str] = set()
        self.score: float = 0.0

    def copy(self):
        new_state = State()
        new_state.assignments = {k: v.copy() for k, v in self.assignments.items()}
        new_state.unassigned_students = self.unassigned_students.copy()
        new_state.score = self.score
        return new_state

class ModifiedMDP:
    def __init__(self, data: dict, gamma: float = 0.9):
        self.data = data
        self.gamma = gamma
        self.students = {s["Nombre"] for s in data["estudiantes"]}
        self.desafios = {d["Titulo"] for d in data["desafios"]}
        self.student_preferences = self._get_student_preferences()
        self.student_careers = self._get_student_careers()
        self.career_limits = self._get_career_limits()

    def _get_student_preferences(self) -> Dict[str, List[str]]:
        preferences = {}
        for student in self.data["estudiantes"]:
            preferences[student["Nombre"]] = student["Postulaciones"]
        return preferences

    def _get_student_careers(self) -> Dict[str, str]:
        return {s["Nombre"]: s["Carrera"] for s in self.data["estudiantes"]}

    def _get_career_limits(self) -> Dict[str, int]:
        return {c["Nombre"]: c["Maximo"] for c in self.data["carreras"]}

    def get_valid_actions(self, state: State) -> List[Tuple[str, str]]:
        actions = []

        for student in state.unassigned_students:
            for desafio in self.desafios:
                if self._is_valid_assignment(state, student, desafio):
                    actions.append((student, desafio))

        return actions

    def _is_valid_assignment(self, state: State, student: str, desafio: str) -> bool:
        if len(state.assignments[desafio]) >= 4:  # Max team size
            return False

        # Check career limits
        student_career = self.student_careers[student]
        career_count = sum(1 for s in state.assignments[desafio]
                          if self.student_careers[s] == student_career)

        if career_count >= self.career_limits[student_career]:
                return False

        return True

    def reward(self, state: State) -> float:
        score = 0.0

        # Preference satisfaction reward
        for desafio, students in state.assignments.items():
            for student in students:
                if desafio in self.student_preferences[student]:
                    pref_index = self.student_preferences[student].index(desafio)
                    score += 1/(1 + pref_index) * 8  # Higher reward for higher preferences
                else:
                    score -= 1

        # Team size penalties/rewards
        for students in state.assignments.values():
            if len(students) < 2:
                score -= 10
            elif len(students) > 4:
                score -= 10

        # Diversity bonus
        for students in state.assignments.values():
            careers = {self.student_careers[s] for s in students}
            score += (1 - 1/(len(careers) + 1))*2

        return score

    def transition(self, state: State, action: Tuple[str, str]) -> State:
        student, desafio = action
        new_state = state.copy()

        new_state.assignments[desafio].append(student)
        new_state.unassigned_students.remove(student)
        new_state.score = self.reward(new_state)

        return new_state

class PolicyIterator:
    def __init__(self, mdp: ModifiedMDP, max_iterations: int = 1_000):
        self.mdp = mdp
        self.max_iterations = max_iterations
        self.min_acceptable_score = 1_000  # Adjust based on your needs

    def find_optimal_assignment(self) -> State:
        initial_state = State()
        initial_state.unassigned_students = set(self.mdp.students)

        current_state = initial_state
        best_state = None
        best_score = float('-inf')

        for current_iteration in range(self.max_iterations):
            if not current_state.unassigned_students:
                score = self.mdp.reward(current_state)
                if score > best_score:
                    best_state = current_state.copy()
                    best_score = score

                if score >= self.min_acceptable_score:
                    break

                # Reset for next iteration
                current_state = initial_state.copy()
                continue

            actions = self.mdp.get_valid_actions(current_state)
            if not actions:
                current_state = initial_state.copy()
                continue

            # Choose action based on expected value
            action = self._select_action(current_state, actions)
            current_state = self.mdp.transition(current_state, action)
        best_state.iterations = current_iteration
        return best_state

    def _select_action(self, state: State, actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        # Combine exploration and exploitation
        if random.random() < 0.2:  # 20% exploration
            return random.choice(actions)

        # Else choose best action based on immediate reward
        best_action = None
        best_value = float('-inf')

        for action in actions:
            next_state = self.mdp.transition(state.copy(), action)
            value = next_state.score

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

import json
with open('body.json', 'r') as file:
    body = json.load(file)
    # Load your JSON data here
estudiantes = body["estudiantes"]
random.shuffle(estudiantes)
data = {
    "estudiantes": estudiantes,
    "desafios": body["desafios"],
    "carreras": body["carreras"]
}

mdp = ModifiedMDP(data)
policy_iterator = PolicyIterator(mdp)
final_state = policy_iterator.find_optimal_assignment()

# Print results
for desafio, students in final_state.assignments.items():
    print(f"\nDesafío: {desafio}")
    print("Estudiantes asignados:")
    for student in students:
        print(f"- {student} ({mdp.student_careers[student]})")
    print(f"Tamaño del equipo: {len(students)}")

print(f"\nScore final: {final_state.score} en {final_state.iterations} iteraciones")
