# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import util

# Student custom class
@dataclass(frozen=True)
class SearchNode:
    state: Any
    parent: Optional["SearchNode"] = None
    action: Optional[Any] = None
    pathCost: float = 0


@dataclass(frozen=True)
class SearchMetrics:
    solved: bool
    expandedNodes: int
    generatedNodes: int
    maximumFrontierSize: int

# Student custom function
_last_search_metrics = SearchMetrics(False, 0, 0, 0)

def getLastSearchMetrics():
    return _last_search_metrics

def _actionsFrom(node):
    actions = []
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    return actions

def _frontierSize(frontier):
    if hasattr(frontier, "heap"):
        return len(frontier.heap)
    return len(frontier.list)

def _recordMetrics(solved, expanded, generated, maximumFrontier):
    global _last_search_metrics
    _last_search_metrics = SearchMetrics(
        solved, expanded, generated, maximumFrontier
    )


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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start = SearchNode(problem.getStartState())
    frontier = util.Stack()
    frontier.push(start)
    discovered = {start.state}
    expanded = 0
    generated = 1
    maximumFrontier = 1

    while not frontier.isEmpty():
        node = frontier.pop()

        if problem.isGoalState(node.state):
            _recordMetrics(True, expanded, generated, maximumFrontier)
            return _actionsFrom(node)

        expanded += 1
        for successor, action, step_cost in problem.getSuccessors(node.state):
            if successor not in discovered:
                discovered.add(successor)
                frontier.push(
                    SearchNode(successor, node, action, node.pathCost + step_cost)
                )
                generated += 1
        maximumFrontier = max(maximumFrontier, _frontierSize(frontier))

    _recordMetrics(False, expanded, generated, maximumFrontier)
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = SearchNode(problem.getStartState())
    frontier = util.Queue()
    frontier.push(start)
    discovered = {start.state}
    expanded = 0
    generated = 1
    maximumFrontier = 1

    while not frontier.isEmpty():
        node = frontier.pop()

        if problem.isGoalState(node.state):
            _recordMetrics(True, expanded, generated, maximumFrontier)
            return _actionsFrom(node)

        expanded += 1
        for successor, action, step_cost in problem.getSuccessors(node.state):
            if successor not in discovered:
                discovered.add(successor)
                frontier.push(
                    SearchNode(successor, node, action, node.pathCost + step_cost)
                )
                generated += 1
        maximumFrontier = max(maximumFrontier, _frontierSize(frontier))

    _recordMetrics(False, expanded, generated, maximumFrontier)
    return []

def _bestFirstSearch(problem: SearchProblem, heuristic: Callable ):
    start = SearchNode(problem.getStartState())
    frontier = util.PriorityQueue()
    frontier.push(start, heuristic(start.state, problem))
    bestCost = {start.state: 0}
    expanded = 0
    generated = 1
    maximumFrontier = 1

    while not frontier.isEmpty():
        node = frontier.pop()
        if node.pathCost != bestCost.get(node.state):
            continue
        if problem.isGoalState(node.state):
            _recordMetrics(True, expanded, generated, maximumFrontier)
            return _actionsFrom(node)

        expanded += 1

        if expanded > 10000:
            util.TimeoutFunctionException()
        for successor, action, step_cost in problem.getSuccessors(node.state):
            new_cost = node.pathCost + step_cost
            if new_cost < bestCost.get(successor, float("inf")):
                bestCost[successor] = new_cost
                child = SearchNode(successor, node, action, new_cost)
                frontier.push(child, new_cost + heuristic(successor, problem))
                generated += 1
        maximumFrontier = max(maximumFrontier, _frontierSize(frontier))

    _recordMetrics(False, expanded, generated, maximumFrontier)
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return _bestFirstSearch(problem, nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return _bestFirstSearch(problem, heuristic)

def greedyBestFirstSearch(problem: SearchProblem, heuristic=nullHeuristic):
    start = SearchNode(problem.getStartState())
    frontier = util.PriorityQueue()
    frontier.push(start, heuristic(start.state, problem))
    discovered = {start.state}
    expanded = 0
    generated = 1
    maximumFrontier = 1

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            _recordMetrics(True, expanded, generated, maximumFrontier)
            return _actionsFrom(node)

        expanded += 1
        for successor, action, step_cost in problem.getSuccessors(node.state):
            if successor not in discovered:
                discovered.add(successor)
                child = SearchNode(successor, node, action, node.pathCost + step_cost )
                frontier.push(child, heuristic(successor, problem))
                generated += 1
        maximumFrontier = max(maximumFrontier, _frontierSize(frontier))

    _recordMetrics(False, expanded, generated, maximumFrontier)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gbfs = greedyBestFirstSearch