import numpy as np
# https://github.com/jkterry1/MA-ALE-paper/blob/master/play_atari.py 
def moveTo(point):
    angle = np.degrees(np.arctan2(point[0], point[1]))%360.0
    if angle < 22 or angle >= 337:
        return 7, [0, 1]
    elif angle >= 22 and angle < 67:
        return 11, [1,1]
    elif angle >= 67 and angle < 112:
        return 10, [1,0]
    elif angle >= 112 and angle < 157:
        return 9, [1,-1]
    elif angle >= 157 and angle < 202:
        return 5, [0,-1]
    elif angle >= 202 and angle < 247:
        return 1, [-1,-1]
    elif angle >= 247 and angle < 292:
        return 2, [-1,0]
    else: # angle >= 292 and angle < 337:
        return 3, [-1,1]


def sidestep(targetCell,surroundings, occupied):
    openSpaces = []
    for i in surroundings:
        if i not in occupied:
            openSpaces.append(i)
    if len(openSpaces) == 0:
        return targetCell
    distance = np.linalg.norm(np.array(openSpaces)-targetCell, axis=1)
    newTarg = openSpaces[np.argmin(distance)]#sidestep to nearest open empty square
    # print(newTarg)
    return newTarg

def checkAdjacent(target, closedSpaces):
    adjacent = [[target[0]+1,target[1]], [target[0]-1,target[1]], [target[0],target[1]+1],[target[0],target[1]-1]]
    available = []
    for i in adjacent:
        if i not in closedSpaces:
            available.append(i)
    if len(available) == 0:
        return False
    distance = np.linalg.norm(available, axis = 1)
    if min(distance) > 1:
        return False
    else:
        return True

def attack(point):
    angle = np.degrees(np.arctan2(point[0], point[1]))%360.0
    if angle < 22 or angle >= 337:
        return 17
    elif angle >= 22 and angle < 67:
        return 20
    elif angle >= 67 and angle < 112:
        return 19
    elif angle >= 112 and angle < 157:
        return 18
    elif angle >= 157 and angle < 202:
        return 16
    elif angle >= 202 and angle < 247:  
        return 13
    elif angle >= 247 and angle < 292:
        return 14
    else: # angle >= 292 and angle < 337:
        return 15


def policy(obs, ag, env):
    # if (env.terminations[ag] or env.truncations[ag]):
    #     return None
    actionArray = []
    for i in range(len(obs)):
        hostiles = obs[i][:,:,4]
        friendly = obs[i][:,:,1]
        # print(np.asarray(friendly).shape)
        friendlyMini = obs[i][:,:,3]
        hostileMini = obs[i][:,:,6]
        agentloc = [6,6]
        surroundings = [[0, 1],[1, 1],[1, 0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]
        actions = 6
        friendlyloc = (np.asarray(np.where(friendly == 1)).T-agentloc).tolist()
        if 1 in hostiles:
            hostileloc = (np.asarray(np.where(hostiles == 1)).T-agentloc).tolist()
            dist = np.linalg.norm(hostileloc, axis = 1)
            closest = hostileloc[np.argmin(dist)]
            occupied = []
            for i in hostileloc:
                occupied.append(i)
            for j in friendlyloc:
                occupied.append(j)

            if np.min(dist) <= 1:
                actions = attack(closest)
            elif np.min(dist) <= 2**(.5) and checkAdjacent(closest, occupied)== False:
                actions = attack(closest)
            else:
                actions, targetCell = moveTo(closest)
                if targetCell in occupied:
                    newTarget = sidestep(targetCell,surroundings, occupied)
                    actions, targetCell = moveTo(newTarget)
        else:
            probableLoc = np.asarray(np.where(friendlyMini == np.amax(friendlyMini))).T
            correction = (np.asarray(np.where(hostileMini == np.amax(hostileMini))).T)[0].tolist()
            hostileMini[correction[0]][correction[1]] = hostileMini[correction[0]][correction[1]] - 1
            hostile = (np.asarray(np.where(hostileMini != 0)).T)
            probableHostileLoc = np.average(hostile, axis = 0)
            Target = (probableHostileLoc-probableLoc).tolist()
            actions, targ = moveTo(Target[0])

            if targ in friendlyloc:
                newTarget = sidestep(targ,surroundings, friendlyloc)
                actions, targ = moveTo(newTarget)
        actionArray.append(actions)
    actionArray = np.asanyarray(actionArray, dtype=np.int32)
    return actionArray