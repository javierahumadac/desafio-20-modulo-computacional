import copy
from collections import defaultdict
import random

class ProposedAssignment:
    def __init__(self, students, challenges, careers):
        self.students = students
        self.challenges = challenges
        self.careers = {career['Nombre']: career['Maximo'] for career in careers}
        self.assignments = defaultdict(list)
        self.student_assignments = {}

    def can_add_to_team(self, challenge, student):
        current_team = self.assignments[challenge]

        # Verificar tamaño máximo del equipo
        if len(current_team) >= 4:
            return False

        # Contar estudiantes por carrera en el equipo actual
        career_count = defaultdict(int)
        for member in current_team:
            career_count[member['Carrera']] += 1

        # Verificar restricciones de carrera según la configuración
        student_career = student['Carrera']
        max_allowed = self.careers.get(student_career, 1)  # Por defecto 1 si no está especificado

        if career_count[student_career] >= max_allowed:
            return False

        return True

    def randomize_lists(self, seed=None):
        """
        Aleatoriza las listas de estudiantes y desafíos.
        Permite establecer una semilla para reproducibilidad.
        """
        if seed is not None:
            random.seed(seed)

        # Aleatorizar orden de estudiantes
        random.shuffle(self.students)

        # Aleatorizar orden de desafíos disponibles para asignación
        """
        challenges_list = [challenge['Titulo'] for challenge in self.challenges]
        random.shuffle(challenges_list)
        self.shuffled_challenges = challenges_list
        """

    def initial_assignment(self):
        # Primera fase: asignar por primera preferencia con orden aleatorio
        students_copy = self.students.copy()
        random.shuffle(students_copy)  # Aleatorizar orden de asignación

        for student in students_copy:
            if len(student['Postulaciones']) > 0:
                # Intentar asignar a cualquiera de sus preferencias en orden aleatorio
                preferences = student['Postulaciones'].copy()
                #random.shuffle(preferences)  # Aleatorizar orden de preferencias

                for preference in preferences:
                    if self.can_add_to_team(preference, student):
                        self.assignments[preference].append(student)
                        self.student_assignments[student['Nombre']] = preference
                        break

    def find_best_team_for_student(self, student, exclude_challenge=None):
        best_team = None
        best_score = float('-inf')

        # Crear lista de desafíos disponibles y aleatorizarla
        available_challenges = [challenge for challenge, team in self.assignments.items()
                              if challenge != exclude_challenge]
        #random.shuffle(available_challenges)

        for challenge in available_challenges:
            team = self.assignments[challenge]
            if len(team) < 4 and self.can_add_to_team(challenge, student):
                # Calcular score basado en preferencias y tamaño del equipo
                score = 0
                if challenge in student['Postulaciones']:
                    score += (3 - student['Postulaciones'].index(challenge)) * 2
                if len(team) == 1:  # Priorizar equipos que necesitan un miembro más
                    score += 3
                if len(team) == 2:  # También bueno para equipos de 2
                    score += 1

                # Añadir componente aleatorio al score
                score += random.random()  # Añade un valor aleatorio entre 0 y 1

                if score > best_score:
                    best_score = score
                    best_team = challenge

        return best_team

    def resolve_single_member_teams(self):
        while True:
            single_member_teams = [(challenge, team) for challenge, team in self.assignments.items()
                                 if len(team) == 1]

            if not single_member_teams:
                break

            # Aleatorizar el orden de procesamiento de equipos solitarios
            random.shuffle(single_member_teams)

            for challenge, team in single_member_teams:
                student = team[0]
                new_team = self.find_best_team_for_student(student, challenge)

                if new_team:
                    # Mover estudiante al nuevo equipo
                    self.assignments[challenge].remove(student)
                    self.assignments[new_team].append(student)
                    self.student_assignments[student['Nombre']] = new_team
                else:
                    # Si no se encuentra equipo, intentar traer otro estudiante
                    potential_teams = [(ch, tm) for ch, tm in self.assignments.items()
                                     if ch != challenge and len(tm) > 2]
                    random.shuffle(potential_teams)  # Aleatorizar orden de búsqueda

                    found_teammate = False
                    for other_challenge, other_team in potential_teams:
                        # Aleatorizar orden de estudiantes potenciales
                        potential_teammates = other_team.copy()
                        random.shuffle(potential_teammates)

                        for potential_teammate in potential_teammates:
                            if self.can_add_to_team(challenge, potential_teammate):
                                other_team.remove(potential_teammate)
                                self.assignments[challenge].append(potential_teammate)
                                self.student_assignments[potential_teammate['Nombre']] = challenge
                                found_teammate = True
                                break
                        if found_teammate:
                            break

            # Limpiar equipos vacíos
            self.assignments = {k: v for k, v in self.assignments.items() if len(v) > 0}

    def force_assign_remaining(self):
        # Obtener y aleatorizar lista de estudiantes sin asignar
        unassigned = [s for s in self.students if s['Nombre'] not in self.student_assignments]
        random.shuffle(unassigned)

        for student in unassigned:
            best_team = self.find_best_team_for_student(student)

            if best_team:
                self.assignments[best_team].append(student)
                self.student_assignments[student['Nombre']] = best_team
            else:
                # Aleatorizar lista de estudiantes restantes
                remaining_unassigned = [s for s in unassigned if s != student and
                                      s['Nombre'] not in self.student_assignments]
                random.shuffle(remaining_unassigned)

                for other_student in remaining_unassigned:
                    # Aleatorizar lista de desafíos disponibles
                    available_challenges = [c['Titulo'] for c in self.challenges]
                    #random.shuffle(available_challenges)

                    for challenge_title in available_challenges:
                        if (self.can_add_to_team(challenge_title, student) and
                            self.can_add_to_team(challenge_title, other_student)):
                            self.assignments[challenge_title].extend([student, other_student])
                            self.student_assignments[student['Nombre']] = challenge_title
                            self.student_assignments[other_student['Nombre']] = challenge_title
                            break

    def run_assignment(self, seed=None):
        """
        Ejecuta el proceso de asignación con una semilla aleatoria opcional
        """
        if seed is not None:
            random.seed(seed)

        self.randomize_lists(seed)
        self.initial_assignment()
        self.resolve_single_member_teams()
        self.force_assign_remaining()
        return self.assignments, self.student_assignments

    def print_assignment_stats(self):
        print("\nEstadísticas de asignación:")
        all_valid = True

        # Convertir assignments a lista y aleatorizar orden de impresión
        challenges_to_print = list(self.assignments.items())
        #random.shuffle(challenges_to_print)

        for challenge, team in challenges_to_print:
            print(f"\nDesafío: {challenge}")
            print(f"Número de estudiantes: {len(team)}")

            if len(team) < 2:
                print("⚠️ ERROR: Equipo con menos de 2 estudiantes")
                all_valid = False
            elif len(team) > 4:
                print("⚠️ ERROR: Equipo con más de 4 estudiantes")
                all_valid = False

            career_count = defaultdict(int)
            for student in team:
                career_count[student['Carrera']] += 1

            print("Distribución por carrera:")
            for career, count in sorted(career_count.items()):
                max_allowed = self.careers.get(career, 1)
                print(f"- {career}: {count} (máximo permitido: {max_allowed})")
                if count > max_allowed:
                    print(f"⚠️ ERROR: Excede el máximo permitido para {career}")
                    all_valid = False

            print("\nEstudiantes en el equipo:")
            # Aleatorizar orden de impresión de estudiantes
            team_members = team.copy()
            #random.shuffle(team_members)
            for student in team_members:
                postulacion_index = (student['Postulaciones'].index(challenge) + 1
                                   if challenge in student['Postulaciones'] else 0)
                preference_text = f"(Preferencia #{postulacion_index})" if postulacion_index > 0 else "(No preferido)"
                print(f"- {student['Nombre']} ({student['Carrera']}) {preference_text}")

        if all_valid:
            print("\n✅ Todas las asignaciones cumplen con las restricciones")
        else:
            print("\n❌ Hay asignaciones que no cumplen con las restricciones")

        # Mostrar estudiantes sin asignar
        unassigned = self.get_unassigned_students()
        if unassigned:
            print("\n⚠️ Estudiantes sin asignar:")
            for student in unassigned:
                print(f"- {student}")

    def get_unassigned_students(self):
        return [s['Nombre'] for s in self.students if s['Nombre'] not in self.student_assignments]
