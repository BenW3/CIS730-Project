"""
First demo, show the usage of API
"""

from examples.models.tf_model import DeepQNetwork
from AngryAntsBot import policy
from magent import render
from PIL import Image
# import numpy as np
import magent
import math
import random

leftID, rightID = 0, 1


def generate_map(env, map_size, handles, num_agents):
    """generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    global leftID, rightID
    leftID, rightID = rightID, leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    standoff1 = random.randint(-(height - side) // 2, (height - side) // 2)
    standoff2 = random.randint(-(height - side) // 2, (height - side) // 2)
    i = 0
    while i < num_agents:
        x = random.randint(width//2 - gap - side, width // 2 - gap - side + side)
        y = random.randint((height - side) // 2 + standoff1, (height - side) // 2 + side + standoff1)
        if [x,y,0] not in pos:
            pos.append([x,y,0])
            i += 1
    
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    i = 0
    while i < num_agents:
        x = random.randint(width // 2 + gap, width // 2 + gap + side)
        y = random.randint((height - side) // 2 + standoff2, (height - side) // 2 + side + standoff2)
        if [x,y,0] not in pos:
            pos.append([x,y,0])
            i += 1

    env.add_agents(handles[rightID], method="custom", pos=pos)


if __name__ == "__main__":
    map_size = 50
    frame_list = []
    env = magent.GridWorld("battle", map_size=map_size)

    redTeam, blueTeam = env.get_handles()
    # init env and agents
    env.reset()
    generate_map(env, map_size, [redTeam, blueTeam], 100)

    # init two models
    model1 = DeepQNetwork(env, redTeam, "battle-l") #learned policy
    model2 = DeepQNetwork(env, blueTeam, "battle-r") #hand coded policy
    

    # load trained model
    model2.load("saveModel", epoch = 1999)
    model1.load("saveModelHandCoded2", epoch = 1999)

    renderer = render.Renderer(env,map_size, "rgb_array")
    totalReward = [0,0]
    done = False
    step_ct = 0
    print("nums: %d vs %d" % (env.get_num(redTeam), env.get_num(blueTeam)))
    while not done:
        # take actions for deers
        obs_1 = env.get_observation(redTeam)
        ids_1 = env.get_agent_id(redTeam)
        acts_1 = model1.infer_action(obs_1, ids_1)
        # acts_1 = policy(obs_1[0], ids_1, env)
        env.set_action(redTeam, acts_1)

        # take actions for tigers
        obs_2 = env.get_observation(blueTeam)
        ids_2 = env.get_agent_id(blueTeam)
        # acts_2 = policy(obs_2[0], ids_2, env)
        acts_2 = model2.infer_action(obs_2, ids_2)
        env.set_action(blueTeam, acts_2)

        # simulate one step
        done = env.step()

        # render
        # renderer.render()
        frame_list.append(Image.fromarray(renderer.render("rgb_array")))

        # get reward
        reward = [sum(env.get_reward(redTeam)), sum(env.get_reward(blueTeam))]
        totalReward[0] += reward[0]
        totalReward[1] += reward[1]
        # clear dead agents
        env.clear_dead()

        # print info
        if step_ct % 10 == 0:
            print(
                "step: %d\t redTeam' reward: %d\t blueTeam' reward: %d"
                % (step_ct, reward[0], reward[1])
            )

        step_ct += 1
        if step_ct > 500:
            break
    print("redTeam reward:"+str(totalReward[0])+"   blueTeam reward:"+str(totalReward[1]) )
    print("rendering gif of length: "+str(len(frame_list)))
    frame_list[0].save('tesDiffVIdent.gif', save_all=True, append_images=frame_list[1:], duration=0.1, loop=0)
