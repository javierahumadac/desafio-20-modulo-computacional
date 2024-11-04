import optuna
from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
from typing import List, Dict, Tuple
from rl.assignation_env import AssignationEnvironment

def ppo_assignment(
        students: List[Dict],
        challanges: List[Dict],
        careers: List[Dict]
    ) -> List[Dict]:
 # 120_000
    def optimize_ppo(n_trials=15, n_timesteps=120_000, n_eval_episodes=50) -> Tuple[dict, float, List[Dict]]:
        """
        Optimiza hiperparámetros de PPO usando Optuna y retorna las mejores asignaciones.

        Args:
            n_trials: Número de pruebas de optimización
            n_timesteps: Pasos de entrenamiento por prueba
            n_eval_episodes: Episodios de evaluación por prueba

        Returns:
            Tuple con los mejores parámetros, mejor valor y las mejores asignaciones
        """
        best_assignments = None

        def objective(trial):
            nonlocal best_assignments

            # Definimos el espacio de búsqueda de hiperparámetros
            model_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 32, 2048),
                "batch_size": trial.suggest_int("batch_size", 32, 256),
                "n_epochs": trial.suggest_int("n_epochs", 5, 20),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.01),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 0.9),
            }

            # Creamos el ambiente y el modelo
            env = AssignationEnvironment(
                students,
                challanges,
                careers
            )
            model = PPO("MlpPolicy", env, verbose=0, **model_params)

            # Entrenamos el modelo
            model.learn(total_timesteps=n_timesteps)

            # Evaluamos el modelo y guardamos las mejores asignaciones
            best_reward = float('-inf')
            best_trial_assignments = None

            for _ in range(n_eval_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_trial_assignments = env.episode_assignments.copy()

            # Actualizamos las mejores asignaciones globales si este trial es el mejor
            if trial.should_prune():
                raise optuna.TrialPruned()

            if best_trial_assignments and (best_assignments is None or best_reward > trial.study.best_value):
                best_assignments = best_trial_assignments

            env.close()
            return best_reward

        # Creamos el estudio de Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Retornamos los mejores parámetros, valor y asignaciones
        return study.best_params, study.best_value, best_assignments

    # Ejecutamos la optimización y retornamos solo las mejores asignaciones
    _, _, best_assignments = optimize_ppo()
    return best_assignments
