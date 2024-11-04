import math
from collections import defaultdict

def calculate_possible_states(n_students):
    """
    Calcula una estimación del número de estados posibles
    considerando las restricciones de tamaño de grupo (2-4 estudiantes)
    """
    # Primero calculamos las posibles formas de particionar los estudiantes
    # en grupos de 2-4

    def calculate_partitions(n, min_size=2, max_size=4, memo=None):
        if memo is None:
            memo = {}

        if n < 0:
            return 0
        if n == 0:
            return 1
        if n in memo:
            return memo[n]

        total = 0
        for size in range(min_size, min(max_size + 1, n + 1)):
            # Multiplicamos por el número de formas de elegir los estudiantes para este grupo
            combinations = math.comb(n, size)
            total += combinations * calculate_partitions(n - size, min_size, max_size, memo)

        memo[n] = total
        return total

    # Calculamos una estimación inicial
    n_partitions = calculate_partitions(n_students)

    print("\nAnálisis de estados posibles:")
    print(f"Número de estudiantes: {n_students}")
    print(f"Particiones posibles (sin considerar restricciones de carrera): {n_partitions:,}")

    # Estimación considerando restricciones de carrera (muy aproximada)
    print("\nEstimación con restricciones:")
    print("- Cada grupo debe tener entre 2 y 4 estudiantes")
    print("- Los grupos deben respetar límites de carrera")

    # Calculamos número mínimo y máximo de grupos posibles
    min_groups = math.ceil(n_students / 4)  # Mínimo número de grupos (todos de tamaño 4)
    max_groups = math.floor(n_students / 2)  # Máximo número de grupos (todos de tamaño 2)

    print(f"\nRango de número de grupos posibles:")
    print(f"- Mínimo: {min_groups} grupos (usando grupos de 4)")
    print(f"- Máximo: {max_groups} grupos (usando grupos de 2)")

    # Calcular una estimación más conservadora
    conservative_estimate = n_partitions // 1000  # Asumiendo que solo 0.1% cumple restricciones de carrera

    print(f"\nEstimación conservadora de estados válidos: {conservative_estimate:,}")
    print("(Considerando restricciones de carrera y tamaño de grupo)")

    # Calcular tiempo estimado de generación
    time_per_state = 0.001  # 1 millisegundo por estado (muy optimista)
    total_time = conservative_estimate * time_per_state

    print("\nTiempo estimado para generar todos los estados:")
    print(f"- Segundos: {total_time:,.2f}")
    print(f"- Minutos: {total_time/60:,.2f}")
    print(f"- Horas: {total_time/3600:,.2f}")
    print(f"- Días: {total_time/86400:,.2f}")

    # Calcular espacio en memoria requerido
    bytes_per_state = n_students * 4  # 4 bytes por estudiante (entero)
    total_memory = conservative_estimate * bytes_per_state

    print("\nEspacio en memoria requerido (estimado):")
    print(f"- Bytes: {total_memory:,}")
    print(f"- Megabytes: {total_memory/1024/1024:,.2f}")
    print(f"- Gigabytes: {total_memory/1024/1024/1024:,.2f}")

    return conservative_estimate

if __name__ == "__main__":
    n_students = 64  # Número de estudiantes en tu caso
    total_states = calculate_possible_states(n_students)
