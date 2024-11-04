import numpy as np
from collections import defaultdict, Counter

def calculate_metrics(students: list, challanges: list, assignments: list, careers: list) -> dict:
    """
    Calcula métricas de calidad para una asignación de estudiantes a desafíos

    Args:
        students: Lista de estudiantes con Nombre,Carrera y Postulaciones
        challanges: Lista de desafíos con Título y Carrerras
        assignments: Lista de desafíos asignados a los estudiantes
        careers: Lista de carreras con Nombre y Máximo

    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    metrics = {}

    # Crear diccionario de equipos
    equipos = defaultdict(list)
    for assignment in assignments:
        equipos[assignment['Desafio']].append(assignment['Nombre'])

    # Diccionario para mapear estudiante a sus postulaciones y carreras
    estudiantes_postulaciones = {s['Nombre']: s['Postulaciones'] for s in students}
    estudiantes_carreras = {s['Nombre']: s['Carrera'] for s in students}

    # Diccionario de límites por carrera
    limites_carreras = {c['Nombre']: c['Maximo'] for c in careers}

    # Calcular satisfacción individual y métricas relacionadas
    satisfacciones = []
    primera_prioridad = 0
    fuera_preferencias = 0

    for assignment in assignments:
        nombre = assignment['Nombre']
        desafio = assignment['Desafio']
        postulaciones = estudiantes_postulaciones[nombre]

        try:
            indice = postulaciones.index(desafio)
            satisfaccion = 1 / (indice + 1)
            if indice == 0:
                primera_prioridad += 1
        except ValueError:
            satisfaccion = 0
            fuera_preferencias += 1

        satisfacciones.append(satisfaccion)

    # Calcular tamaños de equipos y carreras por equipo
    tamanos_equipos = []
    carreras_por_equipo = []
    equipos_por_tamano = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    equipos_exceden_limite = 0

    for desafio, miembros in equipos.items():
        if len(miembros) > 0:  # Solo considerar equipos con al menos un miembro
            tamano = len(miembros)
            tamanos_equipos.append(tamano)

            # Contar tamaños de equipo
            if tamano >= 5:
                equipos_por_tamano[5] += 1
            else:
                equipos_por_tamano[tamano] += 1

            # Contar carreras en el equipo
            carreras_equipo = Counter(estudiantes_carreras[estudiante] for estudiante in miembros)
            carreras_por_equipo.append(len(carreras_equipo))

            # Verificar límites por carrera
            excede_limite = False
            for carrera, cantidad in carreras_equipo.items():
                if cantidad > limites_carreras.get(carrera, 0):
                    excede_limite = True
                    break
            if excede_limite:
                equipos_exceden_limite += 1

    # Calcular desafíos sin equipo
    desafios_totales = set(c['Titulo'] for c in challanges)
    desafios_asignados = set(equipos.keys())
    desafios_sin_equipo = len(desafios_totales - desafios_asignados)

    # Almacenar métricas
    metrics['satisfaccion_promedio'] = np.mean(satisfacciones)
    metrics['estudiantes_primera_prioridad'] = primera_prioridad
    metrics['estudiantes_fuera_preferencias'] = fuera_preferencias
    metrics['estudiantes_sin_desafios'] = len(students) - len(assignments)
    metrics['desafios_sin_equipo'] = desafios_sin_equipo
    metrics['std_tamano_equipos'] = np.std(tamanos_equipos)
    metrics['promedio_carreras_por_equipo'] = np.mean(carreras_por_equipo)
    metrics['total_equipos'] = len([eq for eq in equipos.values() if len(eq) > 0])
    metrics['tamano_promedio_equipo'] = np.mean(tamanos_equipos)
    metrics['equipos_tamano_1'] = equipos_por_tamano[1]
    metrics['equipos_tamano_2'] = equipos_por_tamano[2]
    metrics['equipos_tamano_3'] = equipos_por_tamano[3]
    metrics['equipos_tamano_4'] = equipos_por_tamano[4]
    metrics['equipos_tamano_5_o_mas'] = equipos_por_tamano[5]
    metrics['equipos_exceden_limite_carrera'] = equipos_exceden_limite

    return metrics
