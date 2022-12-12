import re
from turtle import st
import util


class SearchProblem:
    # Returns the start state for the search problem.
    def getStartState(self):
        util.raiseNotDefined()

    # Returns True if and only if the state is a valid goal state.
    def isGoalState(self, state):
        util.raiseNotDefined()

    """
    Return a list of triples (successor,action, stepCost)
        'successor' is a successor to the current state,
        'action' is the action required to get there,
        'stepCost' is the incremental cost of expanding to that successor.
    """
    def getSuccessors(self, state):
        util.raiseNotDefined()

    # Returns the total cost of a particular sequence of actions with legal moves.
    def getCostOfActions(self, actions):
        util.raiseNotDefined()


# Returns a sequence of moves that solves tinyMaze.
def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# Search the deepest nodes in the search tree first.
def depthFirstSearch(problem):
    from util import Stack
    state_to_visit = Stack()
    visited_state = []
    actions_to_current = Stack()
    actions = []
    state = problem.getStartState()

    while not problem.isGoalState(state):
        if state not in visited_state:
            visited_state.append(state)
            for next_state, action, cost in problem.getSuccessors(state):
                temp_action = actions + [action]
                actions_to_current.push(temp_action)
                state_to_visit.push(next_state)
        state = state_to_visit.pop()
        actions = actions_to_current.pop()

    return actions
    util.raiseNotDefined()


# Search the shallowest nodes in the search tree first.
def breadthFirstSearch(problem):
    from util import Queue
    state_to_visit = Queue()
    visited_state = []
    actions = []
    actions_to_current = Queue()
    state = problem.getStartState()

    while not problem.isGoalState(state):
        if state not in visited_state:
            visited_state.append(state)
            for next_state, action, cost in problem.getSuccessors(state):
                temp_action = actions + [action]
                actions_to_current.push(temp_action)
                state_to_visit.push(next_state)
        state = state_to_visit.pop()
        actions = actions_to_current.pop()

    return actions

    util.raiseNotDefined()


# Search the node of least total cost first.
def uniformCostSearch(problem):
    from util import PriorityQueue
    state_to_visit = PriorityQueue()
    visited_state = []
    actions = []
    actions_to_current = PriorityQueue()
    state = problem.getStartState()

    while not problem.isGoalState(state):
        if state not in visited_state:
            visited_state.append(state)
            for next_state, action, cost in problem.getSuccessors(state):
                temp_action = actions + [action]
                cost_of_state = problem.getCostOfActions(temp_action)
                actions_to_current.push(temp_action, cost_of_state)
                state_to_visit.push(next_state, cost_of_state)
        state = state_to_visit.pop()
        actions = actions_to_current.pop()

    return actions
    util.raiseNotDefined()


#  Estimating the cost from the current state to the nearest goal in the provided SearchProblem.
def nullHeuristic(state, problem=None):
    return 0


# Search the node that has the lowest combined cost and heuristic first.
def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    state_to_visit = PriorityQueue()
    visited_state = []
    actions = []
    actions_to_current = PriorityQueue()
    state = problem.getStartState()

    while not problem.isGoalState(state):
        if state not in visited_state:
            visited_state.append(state)
            for next_state, action, cost in problem.getSuccessors(state):
                temp_action = actions + [action]
                cost_of_state = problem.getCostOfActions(temp_action)
                actions_to_current.push(temp_action, cost_of_state + heuristic(next_state, problem))
                state_to_visit.push(next_state, cost_of_state + heuristic(next_state, problem))
        state = state_to_visit.pop()
        actions = actions_to_current.pop()

    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
