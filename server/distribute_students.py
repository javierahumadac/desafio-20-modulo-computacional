from typing import List, Dict
from copy import deepcopy
import random
from collections import Counter


def shuffle_students(students: List[Dict]) -> List[Dict]:
    """Crea una copia desordenada de la lista de estudiantes"""
    shuffled = deepcopy(students)
    random.shuffle(shuffled)
    return shuffled

def sort_students_saturation(students: List[Dict], order: str = "asc") -> List[Dict]:
    """
    Ordena los estudiantes según la saturación promedio de sus desafíos preferidos

    Args:
        students: Lista de estudiantes
        order: "asc" para orden ascendente, "desc" para descendente

    Returns:
        Lista ordenada de estudiantes
    """
    def get_challenge_saturation(challenge: str, students: List[Dict]) -> int:
        """
        Calcula cuántos estudiantes han seleccionado un desafío específico
        """
        return sum(1 for student in students if challenge in student["Postulaciones"])

    def calculate_student_saturation(student: Dict, students: List[Dict]) -> float:
        """
        Calcula el promedio de saturación de los desafíos preferidos por el estudiante
        """
        if not student["Postulaciones"]:
            return 0

        total_saturation = sum(
            get_challenge_saturation(challenge, students)
            for challenge in student["Postulaciones"]
        )
        return total_saturation / len(student["Postulaciones"])

    if order not in ["asc", "desc"]:
        raise ValueError('order debe ser "asc" o "desc"')

    # Calcular saturación para cada estudiante
    student_saturations = {
        student["Nombre"]: calculate_student_saturation(student, students)
        for student in students
    }

    # Ordenar estudiantes
    return sorted(
        students,
        key=lambda x: student_saturations[x["Nombre"]],
        reverse=(order == "desc")
    )

def sort_students_career(students: List[Dict], order: str = "asc") -> List[Dict]:
    """
    Ordena los estudiantes según el tamaño de su carrera

    Args:
        students: Lista de estudiantes
        order: "asc" para orden ascendente, "desc" para descendente

    Returns:
        Lista ordenada de estudiantes
    """
    if order not in ["asc", "desc"]:
        raise ValueError('order debe ser "asc" o "desc"')

    # Contar estudiantes por carrera
    career_sizes = Counter(student["Carrera"] for student in students)

    # Ordenar estudiantes
    return sorted(
        students,
        key=lambda x: career_sizes[x["Carrera"]],
        reverse=(order == "desc")
    )
