# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState : GameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates : list = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if newPos == currentGameState.getPacmanPosition():
            return -1000000

        min_food_dist = 100000
        x = 0
        for row in newFood:
            for y in range(len(row)):
                if row[y]:
                    min_food_dist = min(min_food_dist, manhattanDistance(newPos, (x, y)))
            x += 1
        
        min_food_dist = max(min_food_dist, .05)

        min_ghost_dist = 100000
        for ghost in newGhostStates:
            min_ghost_dist = min(min_ghost_dist, util.manhattanDistance(newPos, ghost.getPosition()))

        if min_ghost_dist == 0:
            return -1000000

        foodEaten = currentGameState.getFood().count() - newFood.count()
        return (1 / min_food_dist) + foodEaten + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def ghostLayerHelper(self, gameState: GameState, depthLeft, ghostNum):
        # check we are at a leaf
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # check if reached the end of this 'ply'
        if ghostNum >= gameState.getNumAgents():
            # if so, we can try recursing back to original
            return self.getActionHelper(gameState, depthLeft - 1)[0]
        
        ghost_actions = gameState.getLegalActions(ghostNum)
        min_score = 1000000

        for action in ghost_actions:
            newGhostState = gameState.generateSuccessor(ghostNum, action)
            min_score = min(min_score, self.ghostLayerHelper(newGhostState, depthLeft, ghostNum + 1))

        return min_score


    def getActionHelper(self, gameState: GameState, depthLeft):
        # check depth is valid
        if depthLeft < 1:
            # if not, end now
            return (self.evaluationFunction(gameState), 'Stop')

        # do one 'ply' of depth
        actions = gameState.getLegalActions(0)
        max_score = -1000000
        max_action = None

        for action in actions:
            newPacmanState = gameState.generateSuccessor(0, action)
            result = self.ghostLayerHelper(newPacmanState, depthLeft, 1)
            
            if max_score < result:
                max_score = result
                max_action = action
        
        return (max_score, max_action)

    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.getActionHelper(gameState, self.depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def pacmanLayerHelper(self, gameState: GameState, depthLeft, alpha, beta):
        # check depth is valid
        if depthLeft < 1:
            # if not, end now
            return (self.evaluationFunction(gameState), 'Stop', alpha, beta)

        # do one 'ply' of depth
        actions = gameState.getLegalActions(0)
        max_score = -1000000
        max_action = None

        for action in actions:
            newPacmanState = gameState.generateSuccessor(0, action)
            result = self.ghostLayerHelper(newPacmanState, depthLeft, alpha, beta, 1)

            if max_score < result:
                max_score = result
                max_action = action

            # alpha beta pruning
            if max_score > beta:
                break

            alpha = max(alpha, max_score)
        
        return (max_score, max_action)

    def ghostLayerHelper(self, gameState: GameState, depthLeft, alpha, beta, ghostNum):
        # check we are at a leaf
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # check if reached the end of this 'ply'
        if ghostNum >= gameState.getNumAgents():
            # if so, we can try recursing back to original
            return self.pacmanLayerHelper(gameState, depthLeft - 1, alpha, beta)[0]
        
        ghost_actions = gameState.getLegalActions(ghostNum)
        min_score = 1000000

        for action in ghost_actions:
            newGhostState = gameState.generateSuccessor(ghostNum, action)
            result = self.ghostLayerHelper(newGhostState, depthLeft, alpha, beta, ghostNum + 1)
            min_score = min(min_score, result)

            # alpha beta pruning
            if min_score < alpha:
                break
            
            beta = min(beta, min_score)

        return min_score

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -1000000
        beta = 1000000
        return self.pacmanLayerHelper(gameState, self.depth, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def pacmanLayerHelper(self, gameState: GameState, depthLeft):
        # check depth is valid
        if depthLeft < 1:
            # if not, end now
            return (self.evaluationFunction(gameState), 'Stop')

        # do one 'ply' of depth
        actions = gameState.getLegalActions(0)
        max_score = -1000000
        max_action = 'Stop'

        for action in actions:
            newPacmanState = gameState.generateSuccessor(0, action)
            result = self.ghostLayerHelper(newPacmanState, depthLeft, 1)

            if max_score < result:
                max_score = result
                max_action = action
        
        return (max_score, max_action)

    def ghostLayerHelper(self, gameState: GameState, depthLeft, ghostNum):
        # check we are at a leaf
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # check if reached the end of this 'ply'
        if ghostNum >= gameState.getNumAgents():
            # if so, we can try recursing back to original
            return self.pacmanLayerHelper(gameState, depthLeft - 1)[0]
        
        ghost_actions = gameState.getLegalActions(ghostNum)
        results = []

        for action in ghost_actions:
            newGhostState = gameState.generateSuccessor(ghostNum, action)
            results.append(self.ghostLayerHelper(newGhostState, depthLeft, ghostNum + 1))

        return sum(results) / len(results)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.pacmanLayerHelper(gameState, self.depth)[1]

###############################################################
# copied from last project

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.STOP:  (0, 0)}

    _directionsAsList = _directions.items()

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


# copied from last project
class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]

# copied from previous project
def findPathToClosestDot(gameState):
    """
    Returns a path (a list of actions) to the closest dot, starting from
    gameState.
    """
    # Here are some useful elements of the startState
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)

    "*** YOUR CODE HERE ***"
    # copied BFS code

    queue = util.Queue()
    visited = set()

    # nodes on the queue are of structure: (state, path)
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        state, path = queue.pop()

        if state in visited:
            continue

        if problem.isGoalState(state):
            return path

        visited.add(state) # check this
        
        for new_state, direction, cost in problem.getSuccessors(state):
            if new_state not in visited:
                queue.push((new_state, path + [direction]))
    
    # util.Exception('no solution')     
    return []

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: used search function to find closest food, used given score
    function, and given food count, plus ghost's scared times. 
    """
    "*** YOUR CODE HERE ***"

    ghostStates : list = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    pos = currentGameState.getPacmanPosition()
    
    food = currentGameState.getFood()
    num_food = max(1, food.count())
    
    min_food_dist = max(1, len(findPathToClosestDot(currentGameState)))
    # max_food_dist = 1
    # x = 0
    # for row in food:
    #     for y in range(len(row)):
    #         if row[y]:
    #             max_food_dist = max(max_food_dist, manhattanDistance(pos, (x, y)))

    return currentGameState.getScore() + (1 / min_food_dist) + (1 / num_food) + sum(scaredTimes) # + (1 / max_food_dist)



    # pos = currentGameState.getPacmanPosition()
    # food = currentGameState.getFood()
    # ghostStates : list = currentGameState.getGhostStates()
    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # "*** YOUR CODE HERE ***"
    # min_food_dist = 100000
    # x = 0
    # for row in food:
    #     for y in range(len(row)):
    #         if row[y]:
    #             min_food_dist = min(min_food_dist, manhattanDistance(pos, (x, y)))
    #     x += 1
    
    # min_food_dist = max(min_food_dist, .05)

    # min_ghost_dist = 100000
    # for ghost in ghostStates:
    #     min_ghost_dist = min(min_ghost_dist, util.manhattanDistance(pos, ghost.getPosition()))

    # if min_ghost_dist == 0:
    #     min_ghost_dist = 1

    # foodEaten = currentGameState.getFood().count() - food.count()
    # # return (1 / min_food_dist) + foodEaten + currentGameState.getScore()
    # print(currentGameState.getScore() + (1 / min_ghost_dist))
    # return currentGameState.getScore() + (1 / min_food_dist) + (1 / food.count())


# Abbreviation
better = betterEvaluationFunction
