from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from mcts import mcts_assignment
from rl.ppo import ppo_assignment
from algoritmos.propuesta import ProposedAssignment
from algoritmos.client import ClientAssignment

from distribute_students import shuffle_students, sort_students_career, sort_students_saturation
from metrics import calculate_metrics


app = FastAPI()
NUM_SIMULATIONS = 1_000

class Carrera(BaseModel):
    Nombre: str
    Maximo: int

class Desafio(BaseModel):
    Titulo: str
    Carreras: List[str]

class Postulacion(BaseModel):
    Nombre: str
    Carrera: str
    Postulaciones: List[str]

class RequestBody(BaseModel):
    estudiantes: List[Postulacion]
    desafios: List[Desafio]
    carreras: List[Carrera]

def process_data(data):

    students = [
        {
            "Nombre": student.Nombre,
            "Carrera": student.Carrera,
            "Postulaciones": student.Postulaciones
        }
        for student in data.estudiantes
    ]

    challenges = [
        {
            "Titulo": challenge.Titulo,
            "Carreras": challenge.Carreras
        }
        for challenge in data.desafios
    ]

    careers = [
        {
            "Nombre": career.Nombre,
            "Maximo": career.Maximo
        }
        for career in data.carreras
    ]
    return students, challenges, careers

# === PPO ===
@app.post("/ppo/shuffle_students")
async def ppo_shuffle_students(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = unsorted_students

        # Procesar con MCTS
        result = ppo_assignment(distributed_students, challenges, careers)
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

# === MONTE CARLO TREE SEARCH ===
@app.post("/mcts/shuffle_students")
async def mcts_shuffle_students(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = unsorted_students

        # Procesar con MCTS
        result = mcts_assignment(
            students=distributed_students,
            challanges=challenges,
            careers=careers,
            assignments=None,  # No hay asignaciones previas
            num_simulations=NUM_SIMULATIONS  # Puedes ajustar este número
        )
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

@app.post("/mcts/sort_students_saturation/sort_students_career")
async def mcts_sort_students_saturation(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = sort_students_saturation(unsorted_students, "asc")

        # Procesar con MCTS
        result = mcts_assignment(
            students=distributed_students,
            challanges=challenges,
            careers=careers,
            assignments=None,  # No hay asignaciones previas
            num_simulations=NUM_SIMULATIONS  # Puedes ajustar este número
        )
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

@app.post("/mcts/sort_students_saturation/desc")
async def mcts_sort_students_saturation(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = sort_students_saturation(unsorted_students, "desc")

        # Procesar con MCTS
        result = mcts_assignment(
            students=distributed_students,
            challanges=challenges,
            careers=careers,
            assignments=None,  # No hay asignaciones previas
            num_simulations=NUM_SIMULATIONS  # Puedes ajustar este número
        )
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

@app.post("/mcts/sort_students_career/asc")
async def mcts_sort_students_career(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = sort_students_career(unsorted_students, "asc")

        # Procesar con MCTS
        result = mcts_assignment(
            students=distributed_students,
            challanges=challenges,
            careers=careers,
            assignments=None,  # No hay asignaciones previas
            num_simulations=NUM_SIMULATIONS  # Puedes ajustar este número
        )
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

@app.post("/mcts/sort_students_career/desc")
async def mcts_sort_students_career(data: RequestBody):
    try:
        # Convertir los modelos Pydantic a diccionarios para la función mcts_assignment
        students, challenges, careers = process_data(data)

        # Distribute students
        unsorted_students = shuffle_students(students)
        distributed_students = sort_students_career(unsorted_students, "desc")

        # Procesar con MCTS
        result = mcts_assignment(
            students=distributed_students,
            challanges=challenges,
            careers=careers,
            assignments=None,  # No hay asignaciones previas
            num_simulations=NUM_SIMULATIONS  # Puedes ajustar este número
        )
        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

# === ALGORITMOS ===
@app.post("/algoritmo/propuesta")
async def algoritmo_propuesta(data: RequestBody):
    try:
        students, challenges, careers = process_data(data)
        # Crear instancia
        assigner = ProposedAssignment(students, challenges, careers)

        # Ejecutar asignación
        assignments, student_assignments = assigner.run_assignment()

        result = [{"Nombre": nombre, "Desafio": desafio} for nombre, desafio in student_assignments.items()]

        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

@app.post("/algoritmo/cliente")
async def algoritmo_propuesta(data: RequestBody):
    try:
        students, challenges, careers = process_data(data)
        # Crear instancia
        assigner = ClientAssignment({
            "estudiantes": students,
            "desafios": challenges,
            "carreras": careers})

        # Ejecutar asignación
        assignments, student_assignments = assigner.form_teams()

        result = [{"Nombre": nombre, "Desafio": desafio} for nombre, desafio in student_assignments.items()]

        return {
            "status": "success",
            "message": "Procesamiento completado",
            "metrics": calculate_metrics(students, challenges, result, careers),
            "results": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error durante el procesamiento: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6969)
