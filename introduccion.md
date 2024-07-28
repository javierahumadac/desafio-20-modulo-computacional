# Spaces

## Box

```python
class gymnasium.spaces.Box(
    low: SupportsFloat | NDArray[Any], 
    high: SupportsFloat | NDArray[Any], 
    shape: Sequence[int] | None = None, 
    dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32, 
    seed: int | np.random.Generator | None = None
)
```

Representa una matriz $A \in \R^{d_1\times d_2 \times ...\times d_N}$, siendo mayormente utilizado para el procesamiento de imagenes.

### Parámetros:
El más importante es el `shape`, el cual indica como es la matriz, aunque es opcional, ya que puede ser definida tambien por los parámetros **obligatorios** de `low` y de `high`.
```python
observation_shape = (600, 800, 3)
```
En este ejemplo se puede ver que es una matriz de $600\times 800\times 3$, que finalmente puede representar una imagen con $3$ canales RGB.
```python
spaces.Box(
    low = np.zeros(self.observation_shape), 
    high = np.ones(self.observation_shape),
    dtype = np.float16
)
```
El dtypes representa los posibles tipos de valores que puede tomar un elemento de la matriz, ver [Numpy Sized aliases](https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases), para tener los *Data Types* posibles.

## Discrete
```python
class gymnasium.spaces.Discrete(
    n: int | np.integer[Any], 
    seed: int | np.random.Generator | None = None, 
    start: int | np.integer[Any] = 0
)
```
Representa un rango de números enteros $A = \{ x\in \Z \ | \  a \leq x \leq b\}$ de entre $a$ y $b$.
### Parámetros
En este caso `n` representa el número máximo que es posible tomar $x$ y `start` es el número mínimo, que si no se define es $0$ por defecto.

## MultiBinary
```python
class gymnasium.spaces.MultiBinary(
    n: NDArray[np.integer[Any]] | Sequence[int] | int, 
    seed: int | np.random.Generator | None = None
)
```
Reresenta una matriz binaria $b=\{ b_i \  |\ i\in \{1,2,...,n \}, b_i \in \{0,1\} \}$ de $n$ dimensiones.

### Parámetros
En este caso `n` representa el *shape* de los elementos del espacio.
Si se define como un número entero `x`, el espacio visible será una lista de binarios de tamaño `x`.
```python
spaces.MultiBinary(5)
>>> array([0, 1, 0, 0, 1], dtype=int8)
```
Si se define como un array de elementos $D = [d_1, d_2, ..., d_n]$ donde $d_i \in \N$ representará un array de array de arrays.
```python
spaces.MultiBinary([3, 2])
>>> array([
        [0, 1],
        [1, 1],
        [0, 0],
    ], dtype=int8)
```
## MultiDiscrete
```python
class gymnasium.spaces.MultiDiscrete(
    nvec: NDArray[np.integer[Any]] | list[int], 
    dtype: str | type[np.integer[Any]] = np.int64, 
    seed: int | np.random.Generator | None = None, 
    start: NDArray[np.integer[Any]] | list[int] | None = None
)
```
