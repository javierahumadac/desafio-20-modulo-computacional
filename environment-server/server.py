from flask import Flask, request, jsonify
import gym
from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
import random

# ENVIRONMENT
class OrganizationEnvironment(gym.Env):
    def __init__(self, preferencias = None, rendering = False):
        super(OrganizationEnvironment, self).__init__()

        self.all_rewards = []
        self.rendering = rendering

        if(preferencias == None):
            self.preferencias = self.get_preferencias()
        else:
            self.preferencias = preferencias

        self.num_estudiantes = len(self.preferencias)
        self.num_desafios = len(self.preferencias[0])

        self.num_preferencias = 0
        for i in self.preferencias:
            for j in i:
                if j > self.num_preferencias:
                    self.num_preferencias = j


        self.state = [[ 0 for _ in range(self.num_desafios)] for _ in range(self.num_estudiantes)]
        self.current_ids = list(range(len(self.preferencias)))
        self.reset()

        # Definición de espacio de observación
        self.observation_space = spaces.MultiDiscrete([2] * (self.num_estudiantes * self.num_desafios) + \
                                                      [self.num_desafios] * (self.num_estudiantes * self.num_desafios))
        # Definición de espacio de acción
        self.action_space = spaces.MultiDiscrete([self.num_estudiantes, self.num_desafios])

    def _set_preferencias(self, preferencias):
        self.preferencias = preferencias

    def get_preferencias(self):
        num_estudiantes = 12
        num_desafios = 9

        preferencias_totales = []

        for i in range(num_estudiantes):
            preferencias_estudiante = [0] * (num_desafios)
            numero_preferencia = 1

            while True:
                desafio = random.randint(0, num_desafios - 1)
                if(preferencias_estudiante[desafio] == 0):
                    preferencias_estudiante[desafio] = numero_preferencia
                    numero_preferencia += 1
                if(numero_preferencia > 5):
                    break
            preferencias_totales.append(preferencias_estudiante)

        if(self.rendering):
            print("\t      Preferencias")
            for index, fila in enumerate(preferencias_totales):
                print(f"Alumno {index}: ",fila)

        return preferencias_totales

    def render(self):
        print("\t\t Desafíos\t\t\t Preferencias")
        for index, fila in enumerate(self.state):
            print(f"Alumno {index}: \t",fila,"\t",self.preferencias[index])

    def reset(self):

        last_observation = self._get_observation()

        self.lives = 3
        self.racha_de_aciertos = 0
        # self.preferencias = self.get_preferencias()
        self.reward_acumulado = 0
        self.num_estudiantes = len(self.preferencias)
        self.num_desafios = len(self.preferencias[0])

        self.num_preferencias = 0
        for i in self.preferencias:
            for j in i:
                if j > self.num_preferencias:
                    self.num_preferencias = j


        # Espacio de observación
        self.state = [[ 0 for _ in range(self.num_desafios)] for _ in range(self.num_estudiantes)]
        self.current_ids = list(range(len(self.preferencias)))

        return last_observation

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        id_estudiante = action[0]
        id_desafio = action[1]

        done = self.check_done()

        if self._chech_possible(id_estudiante, id_desafio):

            self.state[id_estudiante][id_desafio] = 1

            if(not done):
                contador_escogidos = 0
                for estudiante in self.state:
                    for desafio in estudiante:
                        contador_escogidos += desafio

            reward = self.get_reward([id_estudiante, id_desafio])

        # else:
        #     self.lives -= 1
        #     reward = -10 #- (3 - self.lives)

        # if not(self.lives > 0):
        #     done = True
            # reward = -90
        done = self.check_done()
        self.reward_acumulado += reward
        # observation = [self.state, self.preferencias]
        observation = self._get_observation()
        info = {}

        if done:
            self.all_rewards.append(self.reward_acumulado)
            return observation, self.reward_acumulado, done, info
        return observation, reward, done, info


    def _chech_possible(self, id_estudiante, id_desafio):
        # Ver si el estudiante ya esta dentro de un desafío
        total_desafios_asignados = 0
        for desafio_estudiante in self.state[id_estudiante]:
            total_desafios_asignados += desafio_estudiante
        if(total_desafios_asignados > 0):
            return False

        return True

    def _get_observation(self):
        # Aplanar el estado y las preferencias para ajustarse al espacio de observación
        return [self.state[i][j] for i in range(self.num_estudiantes) for j in range(self.num_desafios)] + \
               [self.preferencias[i][j] for i in range(self.num_estudiantes) for j in range(self.num_desafios)]

    def get_reward(self, action):
        id_estudiante = action[0]
        id_desafio = action[1]

        reward = 0
        # Reward por las preferencias
        reward +=   (self.num_preferencias - self.preferencias[id_estudiante][id_desafio] + 1)*2 \
                    if self.preferencias[id_estudiante][id_desafio] != 0 \
                    else -1
        return reward

    def check_done(self):
        # Condición para finalizar el episodio
        contador_de_estudiantes = 0
        for estudiante in self.state:
            for seleccion in estudiante:
                contador_de_estudiantes += seleccion

        if(contador_de_estudiantes == self.num_estudiantes):
            return True

        return False

env = OrganizationEnvironment()



model = PPO.load("ppo_model_v5.zip", env=env)

app = Flask(__name__)



# Definir un endpoint básico
@app.route('/api/v1/saludo', methods=['GET'])
def saludo():
    # Obtener el JSON de la solicitud
    datos = request.get_json()

    # print( datos["0"] )

    # Crear una lista de listas a partir del JSON
    keys = sorted(map(int, datos.keys()))  # Ordenar claves numéricas
    lista_final = [datos[str(key)] for key in keys]
    env.reset()
    env._set_preferencias(lista_final)

    obs = env._get_observation()

    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print("\n\n")
        print("Acción:", action,"-> Reward:", reward)
        env.render()
        if done == True:
            break

    # Devolver una respuesta en formato JSON
    respuesta = {
        'resultado': env.state,
        'mensaje': 'Datos procesados correctamente'
    }

    return jsonify(respuesta)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
