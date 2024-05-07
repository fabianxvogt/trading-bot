import pandas as pd
#from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


import numpy as np
import gymnasium as gym
from gym_trading_env.environments import MultiDatasetTradingEnv
from gym_trading_env.utils.history import   History
import talib as ta

vectorize = False

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def preprocess(df : pd.DataFrame):
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"]/df["close"]
    df["feature_high"] = df["high"]/df["close"]
    df["feature_low"] = df["low"]/df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    upperband, middleband, lowerband = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["feature_lband"] = lowerband/df["close"]
    df["feature_uband"] = upperband/df["close"]
    df.dropna(inplace= True)
    return df

# Create your own reward function with the history object
def reward_function(history: History):
    valuation = history["portfolio_valuation", -1]
    valuation_prior = history["portfolio_valuation", -2]
    close = history["data_close", -1]
    close_prior = history["data_close", -2]

    try:
        portfolio_reward = np.log(valuation / valuation_prior)
    except:
        return 0
    return portfolio_reward
    

def get_env():
    #env_func = gym.make_vec if vectorize else gym.make
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir= 'datasets/1h/*.pkl',
        preprocess= preprocess,
        windows= 5,
        positions = [-8, -2, 0, 2, 8],
        #positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=full SHORT), to +1 (=full LONG) with 0 = no position
#        initial_position = 0, #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        #borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration= 1000,
        episodes_between_dataset_switch = 10,

        dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
        #num_envs = 2 if vectorize else None
        #render_mode='logs'
        )
    if vectorize:
        pass
        #gym.register("MultiDatasetTradingEnv", MultiDatasetTradingEnv)

        #env = make_vec_env("MultiDatasetTradingEnv", env_kwargs={'dataset_dir': 'datasets/*.pkl'})
    else:
        env.unwrapped.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
        env.unwrapped.add_metric('Episode Lenght', lambda history : len(history['position']) )
    return env

if __name__ == "__main__":
    #gym.register("MultiDatasetTradingEnv", MultiDatasetTradingEnv)

    model_name = "models/rl_test_bbands"

    # Train the model
    total_timesteps = 10000
    model = None
    intervals = 500000000000000
    env = get_env()
    absolut_profit = 0
    try:
        model = PPO.load(model_name, env)
    except:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001)

    for i in range(1, intervals):  # Divide training into intervals
        model.learn(total_timesteps=1000, reset_num_timesteps=False)

        # Test the model after each interval
        if i % 10 == 0:
            # Save the trained model
            model.save(model_name)
            total_profit = 0
            num_runs = 10
            for _ in range(num_runs):

                observation, info = env.reset()
                #print(info)
                episode_profit = 0
                done = False
                truncated = False
                while not done and not truncated:
                    action, _states = model.predict(observation)
                    observation, reward, done, truncated, info = env.step(action)
                    #print(observation)
                    episode_profit += reward
                total_profit += episode_profit
                absolut_profit += episode_profit
                avg_profit = total_profit / num_runs
                print(f"Interval {i}/{intervals}, Average Profit per Run: {avg_profit}")
            print(f"Total Profit {absolut_profit}")



    # Load the trained model
    model = PPO.load(model_name)
    env = get_env()
    # Test the model
    observation = env.reset()
    total_profit = 0
    done = False
    truncated = False
    while not done and not truncated:
        action, _states = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_profit += reward

    print(f"Total Profit from Testing: {total_profit}")