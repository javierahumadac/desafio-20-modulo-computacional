import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import copy

class ClientAssignment:
    def __init__(self, data):
        self.students = data['estudiantes']
        self.challenges = data['desafios']
        self.careers = data['carreras']
        self.assignments = {}  # Challenge -> List[Student]
        self.student_assignments = {}  # Student_name -> Challenge
        self.unassigned_students = set(student['Nombre'] for student in self.students)
        self.students_dict = {student['Nombre']: student for student in self.students}
        # Mantener preferencias originales intactas
        self.original_preferences = {
            student['Nombre']: student['Postulaciones'].copy()
            for student in self.students
        }
        # Crear copia de trabajo de las preferencias
        self.current_preferences = {
            student['Nombre']: student['Postulaciones'].copy()
            for student in self.students
        }
        self.assignment_order = []

    def count_applications_per_challenge(self) -> Dict[str, int]:
        """Cuenta las postulaciones de prioridad 1 por desafío."""
        challenge_counts = defaultdict(int)
        for student_name in self.unassigned_students:
            if self.current_preferences[student_name]:  # Si tiene postulaciones pendientes
                # Solo contar la primera prioridad actual
                challenge_counts[self.current_preferences[student_name][0]] += 1
        return challenge_counts

    def get_career_limit(self, career: str) -> int:
        """Obtiene el límite máximo de estudiantes por carrera."""
        if career == "Ingeniería Civil Telemática":
            return 3
        return 1

    def can_add_student_to_challenge(self, student: dict, challenge: str) -> bool:
        """Verifica si un estudiante puede ser añadido a un desafío."""
        # Verificar si el estudiante ya está asignado
        if student['Nombre'] in self.student_assignments:
            return False

        if challenge not in self.assignments:
            return True

        current_team = self.assignments[challenge]

        # Verificar límite de equipo - Estricto máximo de 4
        if len(current_team) >= 4:
            return False

        # Contar estudiantes por carrera en el equipo actual
        career_counts = defaultdict(int)
        for team_member in current_team:
            career_counts[team_member['Carrera']] += 1

        # Verificar límite de carrera
        career_limit = self.get_career_limit(student['Carrera'])
        if career_counts[student['Carrera']] >= career_limit:
            return False

        return True

    def update_student_priorities(self, challenge: str):
        """Actualiza las prioridades de los estudiantes no asignados."""
        for student_name in list(self.unassigned_students):
            if challenge in self.current_preferences[student_name]:
                self.current_preferences[student_name].remove(challenge)

    def assign_student_to_challenge(self, student: dict, challenge: str) -> bool:
        """Asigna un estudiante a un desafío y actualiza los registros."""
        if student['Nombre'] in self.student_assignments:
            return False

        if challenge not in self.assignments:
            self.assignments[challenge] = []
            self.assignment_order.append(challenge)

        if len(self.assignments[challenge]) >= 4:
            return False

        self.assignments[challenge].append(student)
        self.student_assignments[student['Nombre']] = challenge
        self.unassigned_students.discard(student['Nombre'])
        return True

    def get_original_preference_number(self, student_name: str, challenge: str) -> str:
        """Obtiene el número de preferencia original del estudiante para un desafío."""
        original_preferences = self.original_preferences[student_name]
        if challenge in original_preferences:
            return f"Preferencia #{original_preferences.index(challenge) + 1}"
        return "Fuera de preferencias"

    def get_compatible_students_for_challenge(self, challenge: str, students: List[dict]) -> List[dict]:
        """
        Obtiene una lista de estudiantes compatibles para un desafío,
        considerando las restricciones de carrera.
        """
        compatible_students = []
        career_counts = defaultdict(int)  # Llevar el conteo de carreras

        for student in students:
            # Verificar si agregar este estudiante excedería el límite de su carrera
            if career_counts[student['Carrera']] >= self.get_career_limit(student['Carrera']):
                continue

            # Si llegamos aquí, el estudiante es compatible
            compatible_students.append(student)
            career_counts[student['Carrera']] += 1

            # Si ya tenemos 4 estudiantes compatibles, es suficiente
            if len(compatible_students) >= 4:
                break

        return compatible_students

    def print_team_statistics(self):
        """Imprime las estadísticas detalladas de los equipos en orden de asignación."""
        print("\nEstadísticas de asignación:")
        print("==========================")

        for challenge in self.assignment_order:
            team = self.assignments[challenge]

            print(f"\nDesafío: {challenge}")
            print(f"Número de estudiantes: {len(team)}")

            career_distribution = defaultdict(int)
            for student in team:
                career_distribution[student['Carrera']] += 1

            print("Distribución por carrera:")
            for career, count in career_distribution.items():
                max_allowed = self.get_career_limit(career)
                print(f"- {career}: {count} (máximo permitido: {max_allowed})")

            print("Estudiantes en el equipo:")
            for student in team:
                preference = self.get_original_preference_number(student['Nombre'], challenge)
                print(f"- {student['Nombre']} ({student['Carrera']}) ({preference})")

    def validate_assignments(self) -> bool:
        """Valida que todas las asignaciones cumplan las restricciones."""
        total_assigned_students = len(self.student_assignments)
        if total_assigned_students != len(self.students):
            print(f"Error: No todos los estudiantes están asignados. Asignados: {total_assigned_students}, Total: {len(self.students)}")
            return False

        for challenge, team in self.assignments.items():
            # Verificar límites de tamaño de equipo
            if len(team) < 2:  # Mínimo absoluto de 2 estudiantes
                print(f"Error: Equipo con menos de 2 estudiantes en {challenge}")
                return False
            if len(team) > 4:  # Máximo de 4 estudiantes
                print(f"Error: Equipo con más de 4 estudiantes en {challenge}, {len(team)}")
                return False

            # Verificar límites por carrera
            career_counts = defaultdict(int)
            for student in team:
                career_counts[student['Carrera']] += 1
                if career_counts[student['Carrera']] > self.get_career_limit(student['Carrera']):
                    print(f"Error: Demasiados estudiantes de {student['Carrera']} en {challenge}")
                    return False

        # Imprimir advertencia sobre equipos pequeños (no es un error)
        small_teams = [challenge for challenge, team in self.assignments.items() if len(team) < 3]
        if small_teams:
            print("\nAdvertencia: Los siguientes desafíos tienen equipos de 2 integrantes:")
            for challenge in small_teams:
                print(f"- {challenge}")

        return True

    def assign_remaining_students_in_pairs(self):
        """
        Asigna los estudiantes restantes asegurando que siempre se formen equipos
        de al menos 2 estudiantes y máximo 4.
        """
        while len(self.unassigned_students) >= 2:
            remaining_students = [self.students_dict[name] for name in self.unassigned_students]
            found_pair = False

            # Primero intentar completar equipos existentes pequeños
            for challenge, team in self.assignments.items():
                if len(team) < 3:  # Priorizar equipos pequeños
                    career_counts = defaultdict(int)
                    for member in team:
                        career_counts[member['Carrera']] += 1

                    compatible_for_team = []
                    for student in remaining_students:
                        if (career_counts[student['Carrera']] < self.get_career_limit(student['Carrera']) and
                            self.can_add_student_to_challenge(student, challenge)):
                            compatible_for_team.append(student)

                    if len(compatible_for_team) >= 2:
                        # Intentar agregar dos estudiantes compatibles
                        students_to_add = compatible_for_team[:2]
                        for student in students_to_add:
                            if self.can_add_student_to_challenge(student, challenge):
                                self.assign_student_to_challenge(student, challenge)
                        found_pair = True
                        break

            if not found_pair:
                # Intentar crear un nuevo equipo
                for challenge in [c['Titulo'] for c in self.challenges]:
                    if challenge not in self.assignments:
                        career_counts = defaultdict(int)
                        compatible_students = []

                        for student in remaining_students:
                            if career_counts[student['Carrera']] < self.get_career_limit(student['Carrera']):
                                compatible_students.append(student)
                                career_counts[student['Carrera']] += 1
                                if len(compatible_students) >= 2:
                                    # Crear nuevo equipo con los estudiantes compatibles
                                    for compatible_student in compatible_students[:2]:
                                        self.assign_student_to_challenge(compatible_student, challenge)
                                    found_pair = True
                                    break
                        if found_pair:
                            break

            if not found_pair:
                break

        # Manejar estudiantes individuales restantes
        for student_name in list(self.unassigned_students):
            student = self.students_dict[student_name]
            for challenge, team in self.assignments.items():
                if len(team) >= 2 and len(team) < 4:
                    career_counts = defaultdict(int)
                    for member in team:
                        career_counts[member['Carrera']] += 1

                    if (career_counts[student['Carrera']] < self.get_career_limit(student['Carrera']) and
                        self.can_add_student_to_challenge(student, challenge)):
                        self.assign_student_to_challenge(student, challenge)
                        break

    def complete_teams(self):
        """Completa equipos asegurando mínimo 3 estudiantes cuando sea posible y respetando máximo 4."""
        incomplete_teams = [
            challenge for challenge, team in self.assignments.items()
            if len(team) < 3
        ]

        for challenge in incomplete_teams:
            current_team = self.assignments[challenge]

            # Intentar mover estudiantes de equipos grandes
            for other_challenge, other_team in self.assignments.items():
                if other_challenge == challenge or len(current_team) >= 4:
                    continue
                if len(other_team) > 3:  # Solo tomar de equipos que pueden ceder estudiantes
                    for student in other_team[:]:
                        if len(current_team) < 3 and self.can_add_student_to_challenge(student, challenge):
                            other_team.remove(student)
                            del self.student_assignments[student['Nombre']]
                            self.assign_student_to_challenge(student, challenge)
                            if len(current_team) >= 3:
                                break

    def form_teams(self) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
        """Forma los equipos siguiendo la heurística especificada."""
        while self.unassigned_students:
            challenge_counts = self.count_applications_per_challenge()
            if not challenge_counts:
                break

            current_challenge = max(challenge_counts.items(), key=lambda x: x[1])[0]

            priority_students = [
                self.students_dict[student_name] for student_name in self.unassigned_students
                if student_name in self.students_dict and
                self.current_preferences[student_name] and
                self.current_preferences[student_name][0] == current_challenge
            ]

            if len(priority_students) >= 2:
                random.shuffle(priority_students)

                # Mantener conteo de carreras para este equipo
                career_counts = defaultdict(int)
                team_members = []

                # Primero encontrar estudiantes compatibles respetando límites de carrera
                for student in priority_students:
                    if (career_counts[student['Carrera']] < self.get_career_limit(student['Carrera']) and
                        len(team_members) < 4):
                        team_members.append(student)
                        career_counts[student['Carrera']] += 1

                # Solo formar equipo si tenemos al menos 2 estudiantes compatibles
                if len(team_members) >= 2:
                    for student in team_members:
                        self.assign_student_to_challenge(student, current_challenge)

            # Actualizar prioridades de los no asignados
            self.update_student_priorities(current_challenge)

        self.assign_remaining_students_in_pairs()
        self.complete_teams()

        return self.assignments, self.student_assignments
