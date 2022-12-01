from magent2.environments import battle_v4
from time import perf_counter
from PIL import Image
from AngryAntsBot import policy
    
env = battle_v4.env(map_size=45, minimap_mode=True, step_reward=-0.005,
dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
max_cycles=1000, extra_features=False, render_mode = 'rgb_array')

frame_list = []
env.reset()
startTime = perf_counter()

redReward = 0
blueReward = 0

for agent in env.agent_iter():
    if perf_counter() - startTime > 240:
        env.close()
        break
    
    observation, reward, done, trunc,info = env.last(observe=True)
    if 'Blue' in agent.title():
        blueReward += reward
    else:
        redReward += reward
    action = policy(observation, agent, env)
    if (env.terminations[agent] or env.truncations[agent]):
        action = None
    env.step(action)
    frame_list.append(Image.fromarray(env.render()))



print('done running, rendering . . .')

frame_list[0].save('test7(equally matched).gif', save_all=True, append_images=frame_list[1:], duration=0.5, loop=0)
print('Red reward: ' + str(redReward))
print('Blue reward: ' + str(blueReward))