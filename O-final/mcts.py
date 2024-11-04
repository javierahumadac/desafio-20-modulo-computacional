from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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
        available_challanges: Dict,
        incomplete_groups: Dict[str, int] = None,
        career_counts: Dict[str, Dict[str, int]] = None,
        total_students_needed: int = 0
    ):
        # Dict donde key es el desafío y value es la lista con estudiantes dando el desafío
        self.assignments = assignments
        # Ejemplo de estructura:
        # {
        #    "SimCity": ["Matias", "Ignacio"]
        # }

        self.available_challanges = available_challanges # Dict donde key es la carrera y value es la lista de desafíos disponibles
        # Ejemplo de estructura:
        # {
        #    "Ing. Civil Telemática": ["SimCity", "Modelo Acústico"]
        # }

        # Dict donde key es el desafío y value es la cantidad de estudiantes en el grupo
        self.incomplete_groups = incomplete_groups if incomplete_groups is not None else {}
        # Ejemplo de estructura:
        # {
        #    "SimCity": 1
        # }

        # Dict donde:
        # - key externa es el desafío
        # - key interna es la carrera
        # - value es la cantidad de estudiantes de esa carrera en ese desafío
        self.career_counts = career_counts if career_counts is not None else {}
        # Ejemplo de estructura:
        # {
        #     "SimCity": {
        #         "Ing. Civil": 2,
        #         "Arquitectura": 1
        #     }
        # }
        self.total_students_needed = total_students_needed
    def __str__(self):
        return f"Assignments: {self.assignments}"

import math
import random
from copy import deepcopy
from tqdm import tqdm

class Node:
    simulation_counter = 0
    def __init__(
        self,
        config: Config,
        current_student_idx: int = 0,
        state: Optional['State'] = None,
        parent: Optional['Node'] = None,
        is_simulation: bool = False  # Nuevo parámetro para identificar nodos de simulación
        ):
        self.is_simulation = is_simulation
        self.simulation_id = None

        if is_simulation:
            Node.simulation_counter += 1
            self.simulation_id = Node.simulation_counter

        # Configuración global
        self.config = config

        # Índice del estudiante actual que estamos asignando
        self.current_student_idx = current_student_idx

        # Estado actual del nodo
        if state is None:
            # Inicializar estado vacío con todos los desafíos disponibles
            available_challanges = {}
            for career in config.careers:
                available_challanges[career["Nombre"]] = [
                    challenge["Titulo"] for challenge in config.challenges
                ]
            self.state = State({}, available_challanges, {})
            # if not is_simulation:
            #     logger.info("Inicializando nodo raíz con estado vacío")
        else:
            self.state = state

        # Flag para saber si es un nodo terminal
        self.is_terminal: bool = self._check_if_terminal()

        if not self.is_terminal:
            self.current_student = self.config.students[self.current_student_idx]
            self.student_career = self.current_student["Carrera"]

        # Referencia al nodo padre
        self.parent = parent

        # Lista de nodos hijos
        self.children: List[Node] = []

        # Métricas MCTS
        self.visits: int = 0  # Número de veces que se ha visitado este nodo
        self.value: float = 0  # Valor acumulado de las simulaciones

        # Lista de acciones posibles desde este estado
        if not self.is_terminal:
            available_challenges = self.state.available_challanges.get(self.student_career, [])
            valid_challenges = [
                challenge for challenge in available_challenges
                if self._is_action_viable(challenge)
            ]


            # Separar y ordenar por preferencias
            preferred_challenges = []
            other_challenges = []

            student_preferences = self.current_student["Postulaciones"]

            for challenge in valid_challenges:
                if challenge in student_preferences:
                    preference_index = student_preferences.index(challenge)
                    preferred_challenges.append((challenge, preference_index))
                else:
                    other_challenges.append(challenge)

            # Ordenar los desafíos preferidos por índice de preferencia
            preferred_challenges.sort(key=lambda x: x[1])
            preferred_challenges = [challenge for challenge, _ in preferred_challenges]

            # Mezclar aleatoriamente los desafíos no preferidos una sola vez
            random.shuffle(other_challenges)

            # Guardar la lista ordenada de acciones
            self.untried_actions = preferred_challenges + other_challenges

        else:
            self.untried_actions = []

    def apply_action(self, challenge: str) -> 'Node':
        """
        Aplica la acción (asigna el estudiante actual al desafío)
        y retorna un nuevo nodo con el estado actualizado
        """
        # Crear nuevas copias de las estructuras del estado
        new_assignments = self.state.assignments.copy()
        new_available_challanges = {
            career: challenges[:]
            for career, challenges in self.state.available_challanges.items()
        }
        new_incomplete_groups = self.state.incomplete_groups.copy()
        new_career_counts = {
            challenge: counts.copy()
            for challenge, counts in self.state.career_counts.items()
        }

        # 1. Actualizar assignments
        if challenge in new_assignments:
            new_assignments[challenge] = new_assignments[challenge] + [self.current_student["Nombre"]]
        else:
            new_assignments[challenge] = [self.current_student["Nombre"]]

        new_group_size = len(new_assignments[challenge])

        # 2. Actualizar incomplete_groups
        if new_group_size < self.config.min_team_size:
            new_incomplete_groups[challenge] = new_group_size
        else:
            new_incomplete_groups.pop(challenge, None)

        # 3. Actualizar career_counts
        if challenge not in new_career_counts:
            new_career_counts[challenge] = {}

        current_career = self.student_career
        new_career_counts[challenge][current_career] = new_career_counts[challenge].get(current_career, 0) + 1

        # 4. Actualizar available_challanges
        # Si el grupo alcanzó el tamaño máximo, remover el desafío de available_challanges
        if new_group_size >= self.config.max_team_size:
            for career in new_available_challanges:
                if challenge in new_available_challanges[career]:
                    new_available_challanges[career].remove(challenge)
        else:
            # Verificar límites por carrera usando career_counts
            career_max = next(c["Maximo"] for c in self.config.careers
                            if c["Nombre"] == current_career)

            if new_career_counts[challenge][current_career] >= career_max:
                if challenge in new_available_challanges.get(current_career, []):
                    new_available_challanges[current_career].remove(challenge)

        # 5. Actualizar total_students_needed
        new_total_students_needed = self.state.total_students_needed

        if challenge in new_assignments:
            old_group_size = len(new_assignments[challenge]) - 1  # sin contar el nuevo estudiante
            if old_group_size < self.config.min_team_size:
                # Si antes necesitábamos X estudiantes, ahora necesitamos X-1
                new_total_students_needed -= 1
        else:
            # Nuevo grupo incompleto, necesitamos (min_size - 1) más estudiantes
            new_total_students_needed += (self.config.min_team_size - 1)

        # Crear nuevo estado con el career_counts actualizado
        new_state = State(
            new_assignments,
            new_available_challanges,
            new_incomplete_groups,
            new_career_counts,
            new_total_students_needed
        )

        # Crear y retornar nuevo nodo
        new_node = Node(
            config=self.config,
            current_student_idx=self.current_student_idx + 1,
            state=new_state,
            parent=self,
            is_simulation=self.is_simulation
        )

        if self.is_simulation:
            new_node.simulation_id = self.simulation_id  # Heredamos el ID de simulación
        return new_node

    def get_uct_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Calcula el score UCT (Upper Confidence Bound for Trees) del nodo
        """
        if self.visits == 0:
            return float('inf')  # Nodos no visitados tienen prioridad máxima

        # Componente de explotación
        exploitation = self.value / self.visits

        # Componente de exploración
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def get_best_child(self, exploration_constant: float = 1.0) -> 'Node':
        """
        Selecciona el mejor hijo basado en el score UCT
        """
        if not self.children:
            raise ValueError("Node has no children")

        return max(self.children, key=lambda child: child.get_uct_score(exploration_constant))

    def _check_if_terminal(self) -> bool:
        """
        Verifica si el nodo actual es terminal:
        - Si ya asignamos al último estudiante
        - Si no hay acciones válidas para el estudiante actual
        """
        # Si ya procesamos todos los estudiantes
        if self.current_student_idx >= len(self.config.students):
            return True

        # Si no hay desafíos disponibles para el estudiante actual
        current_student = self.config.students[self.current_student_idx]
        career = current_student["Carrera"]
        available_challenges = self.state.available_challanges.get(career, [])

        return len(available_challenges) == 0

    # === FASES PRINCIPALES DEL MCTS ===

    def select(self) -> 'Node':
        """
        Selecciona el mejor nodo para expandir usando UCT.
        Desciende por el árbol hasta encontrar:
        - Un nodo terminal
        - Un nodo no completamente expandido (tiene acciones sin probar)
        """
        current = self

        # Mientras el nodo actual no sea terminal y esté completamente expandido
        while not current.is_terminal and current.is_fully_expanded():
            # Si no hay hijos, no podemos seguir descendiendo
            if not current.children:
                break
            # Seleccionar el mejor hijo según UCT
            current = current.get_best_child()

            # logger.info(f"Selección: descendiendo a nodo con estudiante {current.current_student_idx}")
            # time.sleep(0.6)  # Delay para visualización

        return current

    def expand(self) -> Optional['Node']:
        """
        Toma la primera acción no probada (que será la mejor disponible según las preferencias)
        y crea un nuevo nodo hijo.
        """
        if self.is_terminal or not self.untried_actions:
            return None

        # Tomar la primera acción (la mejor según preferencias)
        action = self.untried_actions.pop(0)  # Usamos pop(0) para mantener el orden

        # logger.info(f"Expandiendo: asignando estudiante {self.current_student['Nombre']} al desafío {action}")
        # time.sleep(0.6)  # Delay para visualización

        # Crear nuevo nodo hijo aplicando la acción
        child_node = self.apply_action(action)

        # Agregar el nuevo nodo a la lista de hijos
        self.children.append(child_node)

        return child_node

    def simulate(self) -> float:
        """
        Realiza una simulación desde el estado actual hasta un estado terminal
        y retorna el valor de la simulación
        """
        # Crear una copia del nodo actual para la simulación
        current = Node(
            config=self.config,
            current_student_idx=self.current_student_idx,
            state=deepcopy(self.state),
            parent=None,
            is_simulation=True
        )


        # Realizar movimientos aleatorios hasta llegar a un estado terminal
        action = current.get_random_action()

        while not current.is_terminal:

            current = current.apply_action(action)

            action = current.get_random_action()
            if action is None:  # No hay acciones válidas disponibles
                break

        # Evaluar el estado final
        score = current.evaluate_state()

        return score

    def backpropagate(self, value: float) -> None:
        """Actualiza las estadísticas del nodo y sus padres."""
        current = self

        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    # === FUNCIONES APOYO PARA FASES PRINCIPALES ===

    def is_fully_expanded(self) -> bool:
        """
        Un nodo está completamente expandido cuando todas sus acciones posibles
        han sido probadas (convertidas en nodos hijos)
        """
        return len(self.untried_actions) == 0

    def get_random_action(self) -> Optional[str]:
        """
        En simulación, usa la primera acción disponible de untried_actions
        """
        if not self.untried_actions:
            return None

        if random.random() < 0.7:  # 70% del tiempo usar primera preferencia
            return self.untried_actions[0]
        return random.choice(self.untried_actions)

    def evaluate_state(self) -> float:
        """
        Evalúa qué tan bueno es el estado actual.
        Considera múltiples factores con diferentes pesos.
        """
        if not self.is_terminal:
            return 0.0

        score = 0.0
        assigned_students = sum(len(team) for team in self.state.assignments.values())

        preference_score = 0.0
        if assigned_students > 0:
            preference_matches = 0
            for challenge, team in self.state.assignments.items():
                for student_name in team:
                    student = next(s for s in self.config.students if s["Nombre"] == student_name)
                    if challenge in student["Postulaciones"]:
                        # Dar más peso a las primeras preferencias
                        preference_index = student["Postulaciones"].index(challenge)
                        preference_weight = 1.0 / (preference_index + 1)
                        preference_matches += preference_weight

            preference_score = preference_matches / assigned_students
            score += preference_score

        return score

    def get_most_visited_child(self) -> Optional['Node']:
        """
        Retorna el hijo más visitado.
        Útil para la selección final de la mejor acción después de completar las simulaciones.
        """
        if not self.children:
            return None

        return max(self.children, key=lambda child: child.visits)

    def get_best_action(self, num_simulations: int = 100_000) -> Optional[str]:
        for _ in range(num_simulations):
            leaf = self.select()
            if not leaf.is_terminal:
                leaf = leaf.expand()
                if leaf is None:
                    continue
            value = leaf.simulate()
            leaf.backpropagate(value)

        # Evaluar todos los hijos disponibles
        if not self.children:
            return None

        # Calcular score para cada hijo considerando múltiples factores
        best_score = float('-inf')
        best_action = None

        for child in self.children:
            # Factor 1: Valor promedio de las simulaciones
            simulation_score = child.value / child.visits if child.visits > 0 else 0

            # Factor 2: Prioridad por preferencias del estudiante
            action = None
            for challenge, students in child.state.assignments.items():
                if challenge not in self.state.assignments or \
                len(students) > len(self.state.assignments.get(challenge, [])):
                    action = challenge
                    break

            if action is None:
                continue

            preference_score = 0
            if action in self.current_student["Postulaciones"]:
                preference_index = self.current_student["Postulaciones"].index(action)
                preference_score = 1.0 / (preference_index + 1)  # Mayor score para primeras preferencias

            # Combinar factores con pesos
            total_score = (
                0.4 * simulation_score +  # 40% peso para resultados de simulación
                0.6 * preference_score   # 60% peso para preferencias
            )

            if total_score > best_score:
                best_score = total_score
                best_action = action

        return best_action

    def _is_action_viable(self, challenge: str) -> bool:
        """
        Verifica si tomar esta acción nos deja en un estado viable.
        Prioriza completar grupos incompletos y evalúa si hay suficientes estudiantes.
        """
        remaining_students = len(self.config.students) - self.current_student_idx - 1

        # Verificar límite de carrera
        if challenge in self.state.career_counts:
            current_career_count = self.state.career_counts[challenge].get(self.student_career, 0)
            career_max = next(c["Maximo"] for c in self.config.careers
                            if c["Nombre"] == self.student_career)
            if current_career_count >= career_max:
                return False

        # Si el desafío ya existe en las asignaciones
        if challenge in self.state.assignments:
            current_group_size = len(self.state.assignments[challenge])

            # Si el grupo tiene 1 o 2 estudiantes, es viable automáticamente
            if current_group_size < self.config.min_team_size:
                return True

            # Si el grupo ya tiene 3+ estudiantes
            # Necesitamos calcular si podemos permitir que crezca más
            total_incomplete = sum(self.state.incomplete_groups.values())
            students_needed = total_incomplete * self.config.min_team_size - sum(self.state.incomplete_groups.values())

            # Solo permitimos añadir a grupos completos si hay suficientes estudiantes
            # para completar todos los incompletos
            return remaining_students >= students_needed

        # Si es un nuevo desafío
        else:
            # Calcular estudiantes necesarios incluyendo este nuevo grupo
            total_incomplete = sum(self.state.incomplete_groups.values()) + 1  # +1 por este nuevo grupo
            students_needed = total_incomplete * self.config.min_team_size - (sum(self.state.incomplete_groups.values()) + 1)

            return remaining_students >= students_needed


def mcts_assignment(
    students: List[Dict],
    challanges: List[Dict],
    careers: List[Dict],
    assignments: Optional[List[Dict]] = None,
    num_simulations: int = 1_000
) -> List[Dict]:
    """
    Asigna los estudiantes a desafíos mediante Monte Carlo Tree Search

    Args:
        students: Lista de estudiantes con Nombre, Carrera y Postulaciones
        challanges: Lista de desafíos con Título
        careers: Lista de carreras con Nombre y Máximo
        assignments: Lista opcional de asignaciones previas

    Returns:
        list: Lista de desafíos asignados a los estudiantes
    """
    # Crear la configuración
    config = Config(
        students=students,
        challenges=challanges,
        careers=careers,
        min_team_size=3,
        max_team_size=4
    )

    # Si hay asignaciones previas, crear el estado inicial
    if assignments:
        initial_assignments = {}
        initial_career_counts = {}
        initial_incomplete_groups = {}

        for assignment in assignments:
            challenge = assignment["Desafio"]
            student_name = assignment["Nombre"]

            # Actualizar assignments
            if challenge in initial_assignments:
                initial_assignments[challenge].append(student_name)
            else:
                initial_assignments[challenge] = [student_name]

            # Actualizar career_counts
            if challenge not in initial_career_counts:
                initial_career_counts[challenge] = {}

            student = next(s for s in students if s["Nombre"] == student_name)
            career = student["Carrera"]
            initial_career_counts[challenge][career] = initial_career_counts[challenge].get(career, 0) + 1

            # Actualizar incomplete_groups
            group_size = len(initial_assignments[challenge])
            if group_size < config.min_team_size:
                initial_incomplete_groups[challenge] = group_size

        # Crear available_challanges inicial
        available_challanges = {}
        for career in careers:
            available_challanges[career["Nombre"]] = [
                challenge["Titulo"] for challenge in challanges
            ]
            # Remover desafíos no disponibles
            for challenge, team in initial_assignments.items():
                if len(team) >= config.max_team_size:
                    if challenge in available_challanges[career["Nombre"]]:
                        available_challanges[career["Nombre"]].remove(challenge)
                else:
                    career_count = initial_career_counts[challenge].get(career["Nombre"], 0)
                    if career_count >= career["Maximo"]:
                        if challenge in available_challanges[career["Nombre"]]:
                            available_challanges[career["Nombre"]].remove(challenge)

        initial_state = State(
            initial_assignments,
            available_challanges,
            initial_incomplete_groups,
            initial_career_counts
        )
    else:
        initial_state = None

    # Crear nodo raíz
    root = Node(config=config, state=initial_state)

    # Ejecutar MCTS para cada estudiante no asignado
    current_node = root
    while not current_node.is_terminal:
        # Obtener la mejor acción para el estudiante actual
        best_action = current_node.get_best_action(num_simulations)
        if best_action is None:
            break

        # logger.info(f"[{current_node.current_student_idx + 1}/{len(students)}] {current_node.current_student['Nombre']} -> {best_action}")
        # Aplicar la acción y movernos al siguiente estado
        current_node = current_node.apply_action(best_action)

    # Convertir el estado final al formato de salida requerido
    result = []
    for challenge, team in current_node.state.assignments.items():
        for student_name in team:
            result.append({
                "Nombre": student_name,
                "Desafio": challenge
            })

    return result
