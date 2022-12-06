"""
Train battle, two models in two processes
"""

import argparse
import logging as log
import math
import time

import numpy as np
from examples.model import ProcessingModel
from examples.models import buffer
import random
from PIL import Image
import magent
from magent.render import Renderer
import matplotlib.pyplot as plt

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

# def generate_map(env, map_size, handles):
#     """generate a map, which consists of two squares of agents"""
#     width = height = map_size
#     init_num = map_size * map_size * 0.04
#     gap = 3

#     global leftID, rightID
#     leftID, rightID = rightID, leftID

#     # left
#     n = init_num
#     side = int(math.sqrt(n)) * 2
#     pos = []
#     for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
#         for y in range((height - side) // 2, (height - side) // 2 + side, 2):
#             pos.append([x, y, 0])
#     env.add_agents(handles[leftID], method="custom", pos=pos)

#     # right
#     n = init_num
#     side = int(math.sqrt(n)) * 2
#     pos = []
#     for x in range(width // 2 + gap, width // 2 + gap + side, 2):
#         for y in range((height - side) // 2, (height - side) // 2 + side, 2):
#             pos.append([x, y, 0])
#     env.add_agents(handles[rightID], method="custom", pos=pos)


def play_a_round(
    env, map_size, handles, models, print_every, k, train=True, render=False, eps=None
):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles, 100)
    frame_list = []
    renderer = Renderer(env, map_size, "rgb_array")
    step_ct = 0
    done = False

    n = len(handles)
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print(f"eps {eps:.2f} number {nums}")
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], "e_greedy", eps, block=False)

        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train:
                alives = env.get_alive(handles[i])
                # store samples in replay buffer (non-blocking)
                models[i].sample_step(rewards, alives, block=False)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            frame_list.append(Image.fromarray(renderer.render("rgb_array")))

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # clear dead agents
        env.clear_dead()

        # check return message of previous called non-blocking function sample_step()
        if args.train:
            for model in models:
                model.check_done()
                # print("done")

        if step_ct % print_every == 0:
            print(
                "step %3d,  nums: %s reward: %s,  total_reward: %s "
                % (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2))
            )

        step_ct += 1
        if step_ct > 100:
            break
    if render:
        print('rendering a round ...')
        frame_list[0].save('train'+ str(k+1) + '.gif', save_all=True, append_images=frame_list[1:], duration=0.1, loop=0)

    sample_time = time.time() - start_time
    print(
        "steps: %d,  total time: %.2f,  step average %.2f"
        % (step_ct, sample_time, sample_time / step_ct)
    )

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=1000, block=False)
            # print("trained: "+str(i))
        for i in range(n):
            # print("fetching: "+str(i))
            total_loss[i], value[i] = models[i].fetch_train()
            # print("fetched: "+str(i))

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l):
        return [round(x, 2) for x in l]

    return round_list(total_loss), nums, round_list(total_reward), round_list(value)


if __name__ == "__main__":
    rewardplot = []
    roundplot = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--render_every", type=int, default=200)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--alg", default="dqn", choices=["dqn", "drqn", "a2c"])
    args = parser.parse_args()
    args.train = True
    # set logger
    buffer.init_logger(args.name)

    # init the game
    env = magent.GridWorld("battle", map_size=args.map_size)
    # env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # sample eval observation set
    eval_obs = [None, None]
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles, 100)
        for i in range(len(handles)):
            eval_obs[i] = buffer.sample_observation(env, handles, 2048, 500)

    # load models
    batch_size = 256
    unroll_step = 8
    target_update = 1200
    train_freq = 5

    if args.alg == "dqn":
        from examples.models.tf_model import DeepQNetwork

        RLModel = DeepQNetwork
        base_args = {
            "batch_size": batch_size,
            "memory_size": 2**20,
            "learning_rate": 1e-4,
            "target_update": target_update,
            "train_freq": train_freq,
        }
    elif args.alg == "drqn":
        from examples.models.tf_model import DeepRecurrentQNetwork

        RLModel = DeepRecurrentQNetwork
        base_args = {
            "batch_size": batch_size / unroll_step,
            "unroll_step": unroll_step,
            "memory_size": 8 * 625,
            "learning_rate": 1e-4,
            "target_update": target_update,
            "train_freq": train_freq,
        }
    elif args.alg == "a2c":
        # see train_against.py to know how to use a2c
        raise NotImplementedError

    # init models
    names = [args.name + "-l", args.name + "-r"]
    models = []

    for i in range(len(names)):
        model_args = {"eval_obs": eval_obs[i]}
        model_args.update(base_args)
        models.append(
            ProcessingModel(
                env, handles[i], names[i], 20000 + i, 1000, RLModel, **model_args
            )
        )

    # load if
    savedir = "saveModel"
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print state info
    print(args)
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = (
            buffer.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05])
            if not args.greedy
            else 0
        )
        loss, num, reward, value = play_a_round(
            env,
            args.map_size,
            handles,
            models,
            k=k,
            train=args.train,
            print_every=50,
            render=args.render or (k + 1) % args.render_every == 0,
            eps=eps,
        )  # for e-greedy

        rewardplot.append(reward)
        roundplot.append(k)

        log.info(
            "round %d\t loss: %s\t num: %s\t reward: %s\t value: %s"
            % (k, loss, num, reward, value)
        )
        print(
            "round time %.2f  total time %.2f\n"
            % (time.time() - tic, time.time() - start)
        )

        # save models
        if ((k + 1) == args.n_round) or ((k+1) == args.n_round//2) or ((k+1) == args.n_round*3//4):
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    for model in models:
        model.quit()

    learner = []
    adversary = []
    for i in rewardplot:
        learner.append(i[0])
    for i in rewardplot:
        adversary.append(i[1])
    plt.plot(roundplot,learner, label = 'learner1')
    plt.plot(roundplot,adversary, label = 'learner2')
    # plt.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.xlabel('round number')
    plt.ylabel('reward')
    plt.title("Model Reward Over Time")
    # plt.legend("learner", "hand coded")
    plt.legend()
    # plt.show()
    plt.savefig('learner_reward_identical')
