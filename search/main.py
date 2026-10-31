import sys

import pacman


DEFAULT_DEMO = [
    "-l", "tinyMaze",
    "-p", "SearchAgent",
    "-a", "fn=bfs",
    "-q",
]


if __name__ == "__main__":
    arguments = sys.argv[1:] or DEFAULT_DEMO
    pacman.runGames(**pacman.readCommand(arguments))
