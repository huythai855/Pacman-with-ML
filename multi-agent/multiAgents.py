from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        foods = new_food.asList()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghost_state.scaredTimer for ghost_state in new_ghost_states]

        nearest_ghost_dis = 1e9
        for ghost_state in new_ghost_states:
            ghost_x, ghost_y = ghost_state.getPosition()
            ghost_x, ghost_y = int(ghost_x), int(ghost_y)
            if not ghost_state.scaredTimer:
                nearest_ghost_dis = min(
                    nearest_ghost_dis, manhattanDistance((ghost_x, ghost_y), new_pos)
                )
        nearest_food_dis = 1e9
        for food in foods:
            nearest_food_dis = min(nearest_food_dis, manhattanDistance(food, new_pos))
        if not foods:
            nearest_food_dis = 0
        return (
            successor_game_state.getScore()
            - 7 / (nearest_ghost_dis + 1)
            - nearest_food_dis / 3
        )


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.minimaxSearch(gameState, agentIndex=0, depth=self.depth)[1]

    def minimaxSearch(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.maximizer(gameState, agentIndex, depth)
        else:
            ret = self.minimizer(gameState, agentIndex, depth)
        return ret

    def minimizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        min_score = 1e9
        min_action = Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.minimaxSearch(
                successor_game_state, next_agent, next_depth
            )[0]
            if new_score < min_score:
                min_score, min_action = new_score, action
        return min_score, min_action

    def maximizer(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        max_score = -1e9
        max_action = Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.minimaxSearch(
                successor_game_state, next_agent, next_depth
            )[0]
            if new_score > max_score:
                max_score, max_action = new_score, action
        return max_score, max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.alphabetaSearch(gameState, 0, self.depth, -1e9, 1e9)[1]

    def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.alphasearch(gameState, agentIndex, depth, alpha, beta)
        else:
            ret = self.betasearch(gameState, agentIndex, depth, alpha, beta)
        return ret

    def alphasearch(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        max_score, max_action = -1e9, Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.alphabetaSearch(
                successor_game_state, next_agent, next_depth, alpha, beta
            )[0]
            if new_score > max_score:
                max_score, max_action = new_score, action
            if new_score > beta:
                return new_score, action
            alpha = max(alpha, max_score)
        return max_score, max_action

    def betasearch(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        min_score, min_action = 1e9, Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.alphabetaSearch(
                successor_game_state, next_agent, next_depth, alpha, beta
            )[0]
            if new_score < min_score:
                min_score, min_action = new_score, action
            if new_score < alpha:
                return new_score, action
            beta = min(beta, min_score)
        return min_score, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        return self.expectimaxsearch(gameState, 0, self.depth)[1]

    def expectimaxsearch(self, game_state, agent_index, depth):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            ret = self.evaluationFunction(game_state), Directions.STOP
        elif agent_index == 0:
            ret = self.maximizer(game_state, agent_index, depth)
        else:
            ret = self.expectation(game_state, agent_index, depth)
        return ret

    def maximizer(self, game_state, agent_index, depth):
        actions = game_state.getLegalActions(agent_index)
        if agent_index == game_state.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agent_index + 1, depth
        max_score, max_action = -1e9, Directions.STOP
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score = self.expectimaxsearch(
                successor_game_state, next_agent, next_depth
            )[0]
            if new_score > max_score:
                max_score, max_action = new_score, action
        return max_score, max_action

    def expectation(self, game_state, agent_index, depth):
        actions = game_state.getLegalActions(agent_index)
        if agent_index == game_state.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agent_index + 1, depth
        exp_score, exp_action = 0, Directions.STOP
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            exp_score += self.expectimaxsearch(
                successor_game_state, next_agent, next_depth
            )[0]
        exp_score /= len(actions)
        return exp_score, exp_action


def betterEvaluationFunction(current_game_state):

    pacman_pos = current_game_state.getPacmanPosition()
    food = current_game_state.getFood()
    foods = food.asList()
    ghost_states = current_game_state.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]

    nearest_ghost_dis = 1e9
    for ghost_state in ghost_states:
        ghost_x, ghost_y = ghost_state.getPosition()
        ghost_x, ghost_y = int(ghost_x), int(ghost_y)
        nearest_ghost_dis = (
            min(nearest_ghost_dis, manhattanDistance((ghost_x, ghost_y), pacman_pos))
            if ghost_state.scaredTimer == 0
            else -10
        )

    nearest_food_dis = 1e9
    for food in foods:
        nearest_food_dis = min(nearest_food_dis, manhattanDistance(food, pacman_pos))
    if not foods:
        nearest_food_dis = 0
    return (
        current_game_state.getScore()
        - 7 / (nearest_ghost_dis + 1)
        - nearest_food_dis / 3
    )


# Abbreviation
better = betterEvaluationFunction
