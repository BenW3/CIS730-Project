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



def loadModels(env, model_names = None, model_directories = None):
    redTeam, blueTeam = env.get_handles()
    models = []
    print(model_names[0])
    if model_names != None:
        model1 = DeepQNetwork(env, redTeam, model_names[0]) 
        model1.load(model_directories[0], epoch = 1999)
        models.append(model1)
        print('loading from '+str(model_directories[0]))
        if len(model_names) > 1:
            model2 = DeepQNetwork(env, blueTeam, model_names[1])
            model2.load(model_directories[1], epoch = 1999)
            models.append(model2)
            print('loading from '+str(model_directories[1]))
    return models



def RunBattle(env,map_size, num_agents, models = None, rendergif = None):
    # print(models)
    if rendergif != None:
        renderer = render.Renderer(env,map_size, "rgb_array")
        frame_list = []
    env.reset()
    redTeam, blueTeam = env.get_handles()
    # init env and agents
    generate_map(env, map_size, [redTeam, blueTeam], num_agents)

    totalReward = [0,0]
    done = False
    step_ct = 0
    # print("nums: %d vs %d" % (env.get_num(redTeam), env.get_num(blueTeam)))
    while not done:
        # take actions for redTeam
        obs_1 = env.get_observation(redTeam)
        ids_1 = env.get_agent_id(redTeam)
        if models != None:
            acts_1 = models[0].infer_action(obs_1, ids_1)
            # print("inferring . . . ")
        else:
            acts_1 = policy(obs_1[0], ids_1, env)
        env.set_action(redTeam, acts_1)

        # take actions for blueTeam
        obs_2 = env.get_observation(blueTeam)
        ids_2 = env.get_agent_id(blueTeam)
        if models != None:
            if len(models) > 1:
                acts_2 = models[1].infer_action(obs_2, ids_2)
            else:
                acts_2 = policy(obs_2[0], ids_2, env)
        else:
            acts_2 = policy(obs_2[0], ids_2, env)
        env.set_action(blueTeam, acts_2)

        # simulate one step
        done = env.step()
        if rendergif != None:
            frame_list.append(Image.fromarray(renderer.render("rgb_array")))

        # get reward
        reward = [sum(env.get_reward(redTeam)), sum(env.get_reward(blueTeam))]
        totalReward[0] += reward[0]
        totalReward[1] += reward[1]
        # clear dead agents
        env.clear_dead()

        step_ct += 1
        if step_ct > 500:
            break
    # print("redTeam reward:"+str(totalReward[0])+"   blueTeam reward:"+str(totalReward[1]) )
    if rendergif != None:
        frame_list[0].save(str(rendergif)+ '.gif', save_all=True, append_images=frame_list[1:], duration=0.1, loop=0)
    # pass
    return(totalReward)


if __name__ == "__main__":
    map_size = 50
    env = magent.GridWorld("battle", map_size=map_size)
    num_agents = 100
    iterations = 100
    reward1 = []
    reward2 = []
    reward3 = []
    BobName = "battle-l"
    BobDirectory = "saveModelHandCoded2"
    WrinkleBrainName = "battle-r"
    WrinkleBrainDirectory = "saveModel"

    match1N = [BobName]
    match1D = [BobDirectory]
    match2N = [WrinkleBrainName]
    match2D = [WrinkleBrainDirectory]
    match3N = [BobName,WrinkleBrainName]
    match3D = [BobDirectory, WrinkleBrainDirectory]

    # redSum = 0
    # blueSum = 0
    # redWins = 0
    # blueWins = 0
    # model = loadModels(env,match1N,match1D)
    # # print(model)
    # for i in range(iterations):
    #     print(i)
    #     if i == iterations-1:
    #         reward1.append(RunBattle(env,map_size,num_agents,models = model, rendergif = "eval1"))
    #     else:
    #         reward1.append(RunBattle(env,map_size,num_agents,models = model))
    # for i in reward1:
    #     redSum += i[0]
    #     blueSum += i[1]
    #     if i[0] > i[1]:
    #         redWins += 1
    #     else: 
    #         blueWins += 1
    #     # print(i)
    # print("Bob's total score: " +str(redSum))
    # print("Bob's average: " +str(redSum/iterations))
    # print("Bob's Wins: " +str(redWins))
    # print("AngryAntsBot's total score: " +str(blueSum))
    # print("AngryAntsBot's average: " + str(blueSum/iterations))
    # print("AngryAntsBot's Wins: " +str(blueWins))

    # redSum = 0
    # blueSum = 0
    # redWins = 0
    # blueWins = 0
    # model = loadModels(env,match2N,match2D)
    # # print(model)
    # for i in range(iterations):
    #     print(i)
    #     if i == iterations-1:
    #         reward2.append(RunBattle(env,map_size,num_agents,models = model, rendergif = "eval2"))
    #     else:
    #         reward2.append(RunBattle(env,map_size,num_agents,models = model))
    # for i in reward2:
    #     redSum += i[0]
    #     blueSum += i[1]
    #     if i[0] > i[1]:
    #         redWins += 1
    #     else: 
    #         blueWins += 1
    # print("WrinkleBrain's total score: " +str(redSum))
    # print("WrinkleBrain's average: " +str(redSum/iterations))
    # print("WrinkleBrain's Wins: " +str(redWins))
    # print("AngryAntsBot's total score: " +str(blueSum))
    # print("AngryAntsBot's average: " + str(blueSum/iterations))
    # print("AngryAntsBot's Wins: " +str(blueWins))

    redSum = 0
    blueSum = 0
    redWins = 0
    blueWins = 0
    model = loadModels(env,match3N,match3D)
    # print(model)
    for i in range(iterations):
        print(i)
        if i == iterations-1:
            reward3.append(RunBattle(env,map_size,num_agents,models = model, rendergif = "eval3"))
        else:
            reward3.append(RunBattle(env,map_size,num_agents,models = model))
    for i in reward3:
        redSum += i[0]
        blueSum += i[1]    
        if i[0] > i[1]:
            redWins += 1
        else: 
            blueWins += 1
    print("WrinkleBrain's total score: " +str(blueSum))
    print("WrinkleBrain's average: " +str(blueSum/iterations))
    print("WrinkleBrain's Wins: " +str(blueWins))
    print("Bob's total score: " +str(redSum))
    print("Bob's average: " + str(redSum/iterations))
    print("Bob's Wins: " +str(redWins))