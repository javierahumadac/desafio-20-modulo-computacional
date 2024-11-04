from gym import Env, spaces
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Config:
    students: List[Dict]
    challenges: List[Dict]
    careers: List[Dict]
    min_team_size: int = 3
    max_team_size: int = 4

class State:
    def __init__(
        self,
        assignments: Dict,
        available_challenges: Dict,
        incomplete_groups: Dict[str, int] = None,
        career_counts: Dict[str, Dict[str, int]] = None,
        total_students_needed: int = 0
    ):
        self.assignments = assignments
        self.available_challenges = available_challenges
        self.incomplete_groups = incomplete_groups if incomplete_groups is not None else {}
        self.career_counts = career_counts if career_counts is not None else {}
        self.total_students_needed = total_students_needed

class AssignationEnvironment(Env):
    def __init__(
        self,
        students: List[Dict],
        challenges: List[Dict],
        careers: List[Dict],
        min_team_size: int = 3,
        max_team_size: int = 4
    ):
        super(AssignationEnvironment, self).__init__()

        self.config = Config(
            students=students,
            challenges=challenges,
            careers=careers,
            min_team_size=min_team_size,
            max_team_size=max_team_size
        )

        # Espacios de observación y acción
        self.observation_space = spaces.Discrete(len(students))
        self.action_space = spaces.Discrete(len(challenges))

        # Estado inicial
        self.state = None
        self.current_student_idx = None
        self.done = False
        self.episode_assignments = []
        self.episode_count = 0

        # Reset inicial
        self.reset()

    def reset(self):
        """Reinicia el environment a su estado inicial."""
        # Inicializar estado vacío con todos los desafíos disponibles
        available_challenges = {}
        for career in self.config.careers:
            available_challenges[career["Nombre"]] = [
                challenge["Titulo"] for challenge in self.config.challenges
            ]

        self.state = State({}, available_challenges, {})
        self.current_student_idx = 0
        self.done = False
        self.episode_assignments = []
        self.episode_count += 1

        return self.current_student_idx

    def get_valid_actions(self) -> List[int]:
        """Obtiene la lista de acciones válidas para el estudiante actual."""
        if self.done or self.current_student_idx >= len(self.config.students):
            return []

        current_student = self.config.students[self.current_student_idx]
        career = current_student["Carrera"]

        valid_challenges = []
        available_challenges = self.state.available_challenges.get(career, [])

        for idx, challenge in enumerate(self.config.challenges):
            challenge_title = challenge["Titulo"]
            if challenge_title in available_challenges and self._is_action_viable(challenge_title):
                valid_challenges.append(idx)

        return valid_challenges

    def normalize_action(self, action: int) -> str:
        """Normaliza la acción del agente al espacio de acciones válidas."""
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return None

        normalized_idx = int(action * len(valid_actions) / self.action_space.n)
        normalized_idx = min(normalized_idx, len(valid_actions) - 1)

        challenge_idx = valid_actions[normalized_idx]
        return self.config.challenges[challenge_idx]["Titulo"]


    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Ejecuta un paso en el environment."""
        if self.done:
            return self.current_student_idx, 0.0, True, {"error": "Episode already done"}

        current_student = self.config.students[self.current_student_idx]

        # Normalizar la acción y obtener el desafío correspondiente
        challenge = self.normalize_action(action)
        if challenge is None:
            self.done = True
            print({
                    "error": "No valid actions available",
                    "current_student": current_student["Nombre"],
                    "current_index":   self.current_student_idx
            })
            return self.current_student_idx, -10.0, True, {
                "error": "No valid actions available",
                "current_student": current_student["Nombre"]
            }

        # Agregar asignación al registro del episodio
        self.episode_assignments.append({
            "Nombre": current_student["Nombre"],
            "Desafio": challenge
        })

        # Aplicar la acción y obtener recompensa
        reward = self._apply_action(challenge)


        # Verificar si hemos terminado
        self.done = self.current_student_idx + 1 >= len(self.config.students)

        if not self.done:
            self.current_student_idx += 1

        info = {
            "assigned_challenge": challenge,
            "current_student": current_student["Nombre"],
            "num_valid_actions": len(self.get_valid_actions()),
            "total_assignments": len(self.state.assignments)
        }

        return self.current_student_idx, reward, self.done, info


    def _is_action_viable(self, challenge: str) -> bool:
        """Verifica si una acción es viable en el estado actual."""
        if self.done or self.current_student_idx >= len(self.config.students):
            return False

        remaining_students = len(self.config.students) - self.current_student_idx - 1
        current_student = self.config.students[self.current_student_idx]
        student_career = current_student["Carrera"]

        # Verificar límite de carrera
        if challenge in self.state.career_counts:
            current_career_count = self.state.career_counts[challenge].get(student_career, 0)
            career_max = next(c["Maximo"] for c in self.config.careers
                            if c["Nombre"] == student_career)
            if current_career_count >= career_max:
                return False

        # Verificar grupos incompletos
        if challenge in self.state.assignments:
            current_group_size = len(self.state.assignments[challenge])

            if current_group_size < self.config.min_team_size:
                return True

            total_incomplete = sum(self.state.incomplete_groups.values())
            students_needed = total_incomplete * self.config.min_team_size - sum(self.state.incomplete_groups.values())

            return remaining_students >= students_needed
        else:
            total_incomplete = sum(self.state.incomplete_groups.values()) + 1
            students_needed = total_incomplete * self.config.min_team_size - (sum(self.state.incomplete_groups.values()) + 1)

            return remaining_students >= students_needed

    def _apply_action(self, challenge: str) -> float:
        """Aplica la acción seleccionada y retorna la recompensa."""
        if self.done or self.current_student_idx >= len(self.config.students):
            return 0.0

        current_student = self.config.students[self.current_student_idx]

        # Actualizar assignments
        if challenge in self.state.assignments:
            self.state.assignments[challenge].append(current_student["Nombre"])
        else:
            self.state.assignments[challenge] = [current_student["Nombre"]]

        # Actualizar career_counts
        if challenge not in self.state.career_counts:
            self.state.career_counts[challenge] = {}

        current_career = current_student["Carrera"]
        self.state.career_counts[challenge][current_career] = \
            self.state.career_counts[challenge].get(current_career, 0) + 1

        # Actualizar incomplete_groups
        group_size = len(self.state.assignments[challenge])
        if group_size < self.config.min_team_size:
            self.state.incomplete_groups[challenge] = group_size
        else:
            self.state.incomplete_groups.pop(challenge, None)

        # Actualizar available_challenges
        if group_size >= self.config.max_team_size:
            for career in self.state.available_challenges:
                if challenge in self.state.available_challenges[career]:
                    self.state.available_challenges[career].remove(challenge)
        else:
            # Verificar límites por carrera
            career_max = next(c["Maximo"] for c in self.config.careers
                            if c["Nombre"] == current_career)
            if self.state.career_counts[challenge][current_career] >= career_max:
                if challenge in self.state.available_challenges.get(current_career, []):
                    self.state.available_challenges[current_career].remove(challenge)

        # Calcular recompensa
        reward = self._calculate_reward(challenge, current_student)

        return reward

    def _calculate_reward(self, challenge: str, student: Dict) -> float:
        """
        Calcula la recompensa basada en múltiples factores.
        """
        reward = 0.0

        # Recompensa por preferencias
        if challenge in student["Postulaciones"]:
            preference_index = student["Postulaciones"].index(challenge)
            preference_reward = 1.0 / (preference_index + 1) * 2  # Mayor recompensa para primeras preferencias
            reward += preference_reward
        else:
            reward -= 0.5  # Penalización por asignar desafío fuera de preferencias

        return reward
