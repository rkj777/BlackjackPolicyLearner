import blackjack
import numpy
from pylab import *

numEpisodes = 1000000
Q = 0.00001*rand(182,2)
#adding values for the terminal state
Q[-1] = [0,0] 

#values to use
epsilonMu = 0
epsilonPi = 0.9 
alpha = 0.4

returnSum = 0.0
blackjack.init()

#returns the probability of taking an action given epsilonpi and enxt state
def policy(action,epsilonPi,nextState):
    greedyAction = numpy.argmax(Q[nextState])
    probGreedy = (1-epsilonPi) + (epsilonPi)/2
    if action == greedyAction:
        return probGreedy
    else:
        return 1- probGreedy

#Sums the policies probalities for a state
def policySum(nextState,epsilonPi):
    sum = 0
    for a in range(2):
        sum += policy(a,epsilonPi,nextState) * Q[nextState,a]
    return sum

#function given to the print policy function 
def policyPrint(state):
    return argmax(Q[state])
      
    
for episodeNum in range(numEpisodes):
    #blackjack.init()
    G = 0
    state = 0
    while state != -1:
        #take action according the the beahaviour policy 
        if rand() <= epsilonMu:
            action = randint(2)
        else:
            action = argmax(Q[state])
        #Do that action 
        result = blackjack.sample(state,action)
        reward = result[0]
        newState = result[1]
        
        #Expected Sarsa 
        Q[state, action] = Q[state, action] + alpha *(reward + policySum(newState,epsilonPi) - Q[state, action])
        
        #update values
        G+= reward
        state = newState
         
    if episodeNum % 10000 == 0 and episodeNum != 0:
        print "Episode: ", episodeNum, "Return: ", G, "Average return: ", returnSum/(episodeNum)
    returnSum = returnSum + G
print "Average return: ", returnSum/numEpisodes

print "Running the deterministic policy"
returnSum = 0.0
for episodeNum in range(numEpisodes):
    G = 0
    state = 0
    while state != -1:
        action = argmax(Q[state])
        
        result = blackjack.sample(state,action)
        reward = result[0]
        newState = result[1]
        
        #update values
        G+= reward
        state = newState
    if episodeNum % 10000 == 0 and episodeNum != 0:
        print "Episode: ", episodeNum, "Return: ", G, "Average return: ", returnSum/(episodeNum)
    returnSum = returnSum + G
print "Average return: ", returnSum/(numEpisodes)
blackjack.printPolicy(policyPrint) 