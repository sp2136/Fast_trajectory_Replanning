import random
from random import *
import numpy as np
import matplotlib.pyplot as plt
import pygame
import timeit

#Authors: Deep Shah(djs504) and Shubham Patel(sp2136)

# Get all the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)

WIDTH = 10  # width of each cell
HEIGHT = 10  # height of each cell

MARGIN = 1  # The margin between each cell


# Class definition for a Node in the A* algorithm
class Node:
    def __init__(self, coordinates, parent, g_val, h_val):
        self.coordinates = coordinates  # Coordinates of the node
        self.parent = parent  # Parent node
        self.g_val = g_val  # Cost from start to current node
        self.h_val = h_val  # Heuristic cost from current node to goal node
        self.f_val = self.g_val + self.h_val  # Total estimated cost

    def __lt__(self, other):
        return self.f_val < other.f_val or (self.f_val == other.f_val and self.g_val < other.g_val)


# Define a custom comparison function for nodes in the priority queue
def compare_nodes(node1, node2, g):
    if node1.f_val == node2.f_val:
        if g == "Large":
            return node1.g_val > node2.g_val
        else:
            return node1.g_val < node2.g_val  # Break ties by smallest g-value
    return node1.f_val < node2.f_val


# Create Binary heap methods
def heappush(heap, node, g):
    heap.append(node)
    _siftdown(heap, 0, len(heap) - 1, g)


def heappop(heap, g):
    last_element = heap.pop()
    if heap:
        popped_node = heap[0]
        heap[0] = last_element
        _siftup(heap, 0, g)
        return popped_node
    return last_element


def heapify(heap, g):
    n = len(heap)
    for i in reversed(range(n // 2)):
        _siftup(heap, i, g)


def _siftdown(heap, startpos, pos, g):
    new_item = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if compare_nodes(new_item, parent, g):
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = new_item


def _siftup(heap, pos, g):
    endpos = len(heap)
    startpos = pos
    new_item = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not compare_nodes(heap[childpos], heap[rightpos], g):
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = new_item
    _siftdown(heap, startpos, pos, g)

# Function to compute the heuristic distance between two points (Manhattan distance)
def heuristic(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])


# Function to check if a coordinate is within the bounds of the maze and not blocked
def isValid(currentCoordinates, maze, sizeOfGrid):
    return 0 <= currentCoordinates[0] < sizeOfGrid and 0 <= currentCoordinates[1] < sizeOfGrid and maze[currentCoordinates[0]][currentCoordinates[1]] != 1


# Function to check if a coordinate is in the closed list
def isClosed(currentCoordinates, closedList):
    return currentCoordinates in closedList


# Function to check if a coordinate is in the open list
def isOpen(currentCoordinates, openList):
    for node in openList:
        if node.coordinates == currentCoordinates:
            return openList.index(node)
    return -1


# Function to draw the maze
def draw_maze(window, maze):
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            color = WHITE if maze[row][col] == 0 else BLACK
            pygame.draw.rect(window, color, [(MARGIN + WIDTH) * col + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
    pygame.display.update()


# Function to draw the path with a specified color
def draw_path(window, path, color):
    for node in path:
        pygame.draw.rect(window, color, [(MARGIN + WIDTH) * node[1] + MARGIN, (MARGIN + HEIGHT) * node[0] + MARGIN, WIDTH, HEIGHT])
    pygame.display.update()


# A* algorithm implementation
def a_star(maze, start, target, sizeOfGrid, numberOfExpandedNodes):
    g = "Small"
    openList = []  # Priority queue (heap)
    closedList = set()  # Set to keep track of visited nodes
    openDict = {}  # Dictionary to store nodes in the open list

    # Initialize beginning node and add it to the open list and dictionary
    beginningNode = Node(start, None, 0, heuristic(start, target))
    heappush(openList, beginningNode, g)
    openDict[start] = beginningNode

    while openList:
        # Pop node with lowest f value from the open list
        currNode = openList.pop(0)
        del openDict[currNode.coordinates]  # Remove popped node from open dictionary

        # Add current node to closed list
        closedList.add(currNode.coordinates)

        # Increment number of expanded nodes
        numberOfExpandedNodes += 1

        # If current node is the goal, trace the path and return
        if currNode.coordinates == target:
            plannedPath = []
            currentNode = currNode
            while currentNode is not None:
                plannedPath.append(currentNode.coordinates)
                currentNode = currentNode.parent
            plannedPath.reverse()  # Reverse the path to get it from start to goal
            return plannedPath, numberOfExpandedNodes

        # Generate neighbors
        neighbors = [(currNode.coordinates[0] - 1, currNode.coordinates[1]),
                     (currNode.coordinates[0] + 1, currNode.coordinates[1]),
                     (currNode.coordinates[0], currNode.coordinates[1] - 1),
                     (currNode.coordinates[0], currNode.coordinates[1] + 1)]

        # Iterate over neighbors
        for neighbor in neighbors:
            # Check if neighbor is within bounds and not blocked
            if isValid(neighbor, maze, sizeOfGrid):
                # Calculate tentative g value for neighbor
                tentative_g = currNode.g_val + 1

                # If neighbor is already visited, skip it
                if neighbor in closedList:
                    continue

                # Check if neighbor is already in open list
                if neighbor in openDict:
                    # Compare g values and f values
                    if tentative_g < openDict[neighbor].g_val:
                        openDict[neighbor].g_val = tentative_g
                        openDict[neighbor].f_val = tentative_g + openDict[neighbor].h_val
                        openDict[neighbor].parent = currNode
                        # Re-heapify open list after modifying node values
                        heapify(openList, g)
                else:
                    # Add neighbor to open list and dictionary
                    neighbor_node = Node(neighbor, currNode, tentative_g, heuristic(neighbor, target))
                    heappush(openList, neighbor_node, g)
                    openDict[neighbor] = neighbor_node

    # If no path found, return empty list and number of expanded nodes
    return [], numberOfExpandedNodes


# Repeated A* algorithm implementation
def repeated_a_star(knowledgeMaze, trueMaze, start, target, sizeOfGrid, window):
    plannedPaths = []  # List to store planned paths
    knowledgeMazes = []  # List to store knowledge mazes
    numberOfExpandedNodes = 0  # Counter for expanded nodes
    currentKnowledgeMaze = knowledgeMaze  # Initialize current knowledge maze

    while True:
        # Plan path using A* algorithm
        currentPath, numberOfExpandedNodes = a_star(currentKnowledgeMaze, start, target, sizeOfGrid, numberOfExpandedNodes)
        plannedPaths.append(currentPath)  # Store planned path

        if not currentPath:
            # print("Here")
            return [plannedPaths, knowledgeMazes], numberOfExpandedNodes  # If no path found, return planned paths and knowledge mazes

        # Draw path in RED/MAGENTA color
        # print("coloring")
        if target == (0, 0):
            draw_path(window, currentPath, MAGENTA)
        else:
            draw_path(window, currentPath, RED)

        # Update knowledge maze based on encountered obstacles
        for index, w in enumerate(currentPath):
            if trueMaze[w[0]][w[1]] == 1:
                start = currentPath[index - 1]  # Update start point to coordinate before obstacle
                break
            if w == target:
                return [plannedPaths, knowledgeMazes], numberOfExpandedNodes  # If reached target, return planned paths and knowledge mazes

            # Update knowledge maze with obstacles encountered
            currentCoordinate = currentPath[index]
            neighbors = [(currentCoordinate[0] - 1, currentCoordinate[1]),
                         (currentCoordinate[0] + 1, currentCoordinate[1]),
                         (currentCoordinate[0], currentCoordinate[1] - 1),
                         (currentCoordinate[0], currentCoordinate[1] + 1)]
            for neighbor in neighbors:
                if isValid(neighbor, trueMaze, sizeOfGrid) and trueMaze[neighbor[0]][neighbor[1]] == 1:
                    currentKnowledgeMaze[neighbor[0]][neighbor[1]] = 1

        # Draw knowledge maze
        draw_maze(window, currentKnowledgeMaze)


# Forward A* algorithm which favors large g values
def a_star_with_large_g(maze, start, target, sizeOfGrid, numberOfExpandedNodes):
    g = "Large"
    openList = []  # Priority queue (heap)
    closedList = set()  # Set to keep track of visited nodes
    openDict = {}  # Dictionary to store nodes in the open list

    # Initialize beginning node and add it to the open list and dictionary
    beginningNode = Node(start, None, 0, heuristic(start, target))
    heappush(openList, beginningNode, g)
    openDict[start] = beginningNode

    while openList:
        # Pop node with lowest f value from the open list
        currNode = openList.pop(0)
        del openDict[currNode.coordinates]  # Remove popped node from open dictionary

        # Add current node to closed list
        closedList.add(currNode.coordinates)

        # Increment number of expanded nodes
        numberOfExpandedNodes += 1

        # If current node is the goal, trace the path and return
        if currNode.coordinates == target:
            plannedPath = []
            currentNode = currNode
            while currentNode is not None:
                plannedPath.append(currentNode.coordinates)
                currentNode = currentNode.parent
            plannedPath.reverse()  # Reverse the path to get it from start to goal
            return plannedPath, numberOfExpandedNodes

        # Generate neighbors
        neighbors = [(currNode.coordinates[0] - 1, currNode.coordinates[1]),
                     (currNode.coordinates[0] + 1, currNode.coordinates[1]),
                     (currNode.coordinates[0], currNode.coordinates[1] - 1),
                     (currNode.coordinates[0], currNode.coordinates[1] + 1)]

        # Iterate over neighbors
        for neighbor in neighbors:
            # Check if neighbor is within bounds and not blocked
            if isValid(neighbor, maze, sizeOfGrid):
                # Calculate tentative g value for neighbor
                tentative_g = currNode.g_val + 1

                # If neighbor is already visited, skip it
                if neighbor in closedList:
                    continue

                # Check if neighbor is already in open list
                if neighbor in openDict:
                    # Compare g values and f values
                    if tentative_g > openDict[neighbor].g_val:
                        openDict[neighbor].g_val = tentative_g
                        openDict[neighbor].f_val = tentative_g + openDict[neighbor].h_val
                        openDict[neighbor].parent = currNode
                        # Re-heapify open list after modifying node values
                        heapify(openList, g)
                else:
                    # Add neighbor to open list and dictionary
                    neighbor_node = Node(neighbor, currNode, tentative_g, heuristic(neighbor, target))
                    heappush(openList, neighbor_node, g)
                    openDict[neighbor] = neighbor_node

    # If no path found, return empty list and number of expanded nodes
    return [], numberOfExpandedNodes


# Repeated Forward A* Algorithm
def repeated_a_star_with_large_g(knowledgeMaze, trueMaze, start, target, sizeOfGrid, window):
    plannedPaths = []  # List to store planned paths
    knowledgeMazes = []  # List to store knowledge mazes
    numberOfExpandedNodes = 0  # Counter for expanded nodes
    currentKnowledgeMaze = knowledgeMaze  # Initialize current knowledge maze

    while True:
        # Plan path using A* algorithm
        currentPath, numberOfExpandedNodes = a_star_with_large_g(currentKnowledgeMaze, start, target, sizeOfGrid, numberOfExpandedNodes)
        plannedPaths.append(currentPath)  # Store planned path

        if not currentPath:
            # print("Here")
            return [plannedPaths, knowledgeMazes], numberOfExpandedNodes  # If no path found, return planned paths and knowledge mazes

        # Draw path in RED/MAGENTA color
        # print("coloring")
        if target == (0, 0):
            draw_path(window, currentPath, YELLOW)
        else:
            draw_path(window, currentPath, BLUE)

        # Update knowledge maze based on encountered obstacles
        for index, w in enumerate(currentPath):
            if trueMaze[w[0]][w[1]] == 1:
                start = currentPath[index - 1]  # Update start point to coordinate before obstacle
                break
            if w == target:
                return [plannedPaths, knowledgeMazes], numberOfExpandedNodes  # If reached target, return planned paths and knowledge mazes

            # Update knowledge maze with obstacles encountered
            currentCoordinate = currentPath[index]
            neighbors = [(currentCoordinate[0] - 1, currentCoordinate[1]),
                         (currentCoordinate[0] + 1, currentCoordinate[1]),
                         (currentCoordinate[0], currentCoordinate[1] - 1),
                         (currentCoordinate[0], currentCoordinate[1] + 1)]
            for neighbor in neighbors:
                if isValid(neighbor, trueMaze, sizeOfGrid) and trueMaze[neighbor[0]][neighbor[1]] == 1:
                    currentKnowledgeMaze[neighbor[0]][neighbor[1]] = 1

        # Draw knowledge maze
        draw_maze(window, currentKnowledgeMaze)


# Adaptive A* algorithm implementation
def adaptive_a_star(maze, start, target, sizeOfGrid, numberOfExpandedNodes):
    g = ""
    openList = []  # Priority queue (heap)
    closedList = set()  # Set to keep track of visited nodes
    beginningNode = Node(start, None, 0, heuristic(start, target))
    openList.append(beginningNode)

    while openList:
        currNode = heappop(openList, g)
        closedList.add(currNode.coordinates)
        numberOfExpandedNodes += 1

        if currNode.coordinates == target:
            plannedPath = []
            currentNode = currNode
            while currentNode is not None:
                plannedPath.append(currentNode.coordinates)
                currentNode = currentNode.parent
            plannedPath.reverse()  # Reverse the path to get it from start to goal
            return plannedPath, numberOfExpandedNodes

        neighbors = [(currNode.coordinates[0] - 1, currNode.coordinates[1]),
                     (currNode.coordinates[0] + 1, currNode.coordinates[1]),
                     (currNode.coordinates[0], currNode.coordinates[1] - 1),
                     (currNode.coordinates[0], currNode.coordinates[1] + 1)]

        for neighbor in neighbors:
            if isValid(neighbor, maze, sizeOfGrid):
                if neighbor not in closedList:
                    tentative_g = currNode.g_val + 1
                    neighbor_index = isOpen(neighbor, openList)
                    if neighbor_index != -1:
                        if tentative_g < openList[neighbor_index].g_val:
                            openList[neighbor_index].g_val = tentative_g
                            openList[neighbor_index].f_val = tentative_g + openList[neighbor_index].h_val
                            openList[neighbor_index].parent = currNode
                    else:
                        neighbor_node = Node(neighbor, currNode, tentative_g, heuristic(start, target) - heuristic(start, neighbor))
                        heappush(openList, neighbor_node, g)

    return [], numberOfExpandedNodes


# Repeated Adaptive A* algorithm implementation
def repeated_adaptive_a_star(knowledgeMaze, trueMaze, start, target, sizeOfGrid, window):
    plannedPaths = []
    knowledgeMazes = []
    numberOfExpandedNodes = 0
    currentKnowledgeMaze = knowledgeMaze

    while True:
        currentPath, numberOfExpandedNodes = adaptive_a_star(currentKnowledgeMaze, start, target, sizeOfGrid, numberOfExpandedNodes)
        plannedPaths.append(currentPath)

        if not currentPath:
            return [plannedPaths, knowledgeMazes], numberOfExpandedNodes

        draw_path(window, currentPath, CYAN)

        for index, w in enumerate(currentPath):
            if trueMaze[w[0]][w[1]] == 1:
                start = currentPath[index - 1]
                break
            if w == target:
                return [plannedPaths, knowledgeMazes], numberOfExpandedNodes

            currentCoordinate = currentPath[index]
            neighbors = [(currentCoordinate[0] - 1, currentCoordinate[1]),
                         (currentCoordinate[0] + 1, currentCoordinate[1]),
                         (currentCoordinate[0], currentCoordinate[1] - 1),
                         (currentCoordinate[0], currentCoordinate[1] + 1)]
            for neighbor in neighbors:
                if isValid(neighbor, trueMaze, sizeOfGrid) and trueMaze[neighbor[0]][neighbor[1]] == 1:
                    currentKnowledgeMaze[neighbor[0]][neighbor[1]] = 1

        draw_maze(window, currentKnowledgeMaze)


# Function to generate gridworld using depth-first search approach with random tie-breaking
def generate_gridworld(size=101, num_grids=50):
    for grid_num in range(1, num_grids + 1):
        grid = np.zeros((size, size), dtype=int)  # Initialize grid with all cells unvisited

        # Randomly select initial and final cell locations
        start_row, start_col = (0,0)


        grid[start_row][start_col] = 10  # Mark initial cell


        stack = [(start_row, start_col)]  # Stack for DFS traversal
        visited = set()  # Set to keep track of visited cells

        while stack:
            current_row, current_col = stack[-1]  # Get current cell
            visited.add((current_row, current_col))  # Mark current cell as visited

            # Get unvisited neighbors
            neighbors = [(current_row - 1, current_col), (current_row + 1, current_col),
                         (current_row, current_col - 1), (current_row, current_col + 1)]
            unvisited_neighbors = [(row, col) for row, col in neighbors if 0 <= row < size and 0 <= col < size
                                   and (row, col) not in visited]

            if unvisited_neighbors:  # If unvisited neighbors exist
                next_row, next_col = choice(unvisited_neighbors)  # Randomly select a neighbor
                stack.append((next_row, next_col))  # Add neighbor to stack
                if random() < 0.3:  # 30% probability to mark as blocked
                    grid[next_row][next_col] = 1  # Mark as blocked
                else:
                    grid[next_row][next_col] = 0  # Mark as unblocked
            else:
                stack.pop()  # Backtrack if no unvisited neighbors
        end_row, end_col = (size-1, size-1)
        grid[end_row][end_col] = 5  # Mark final cell
        filename = f'grid{grid_num}.txt'
        np.savetxt(filename, grid, delimiter=",", fmt='%i')


# Generate gridworlds
generate_gridworld()

# Creating the whole grid
num = int(input('Enter a world from 1 to 50: \n'))
if num > 50 or num < 0:
    print("Please enter a number between 0 and 49")
string = 'grid' + str(num) + '.txt'
text_file = open(string, 'r')
lines = text_file.readlines()
grid = []
for line in lines:
    line = line.strip().split(',')
    line = list(map(int, line))
    grid.append(line)

start = (0, 0)
target = (len(grid) - 1, len(grid) - 1)
ROWS = len(grid)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode(((ROWS * 11) + 1, (ROWS * 11) + 1))
pygame.display.set_caption("Fast Trajectory Replanning")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()
time = 0
distance = 0
# Creating the screen
for row in range(ROWS):
    for column in range(ROWS):
        color = WHITE
        if grid[row][column] == 1:
            color = BLACK
        pygame.draw.rect(screen, color,
                         [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])

        if grid[row][column] == 10:
            color = RED
            pygame.draw.rect(screen, color,
                             [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH,
                              HEIGHT])
        if grid[row][column] == 5:
            color = GREEN
            pygame.draw.rect(screen, color,
                             [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH,
                              HEIGHT])

pygame.display.flip()

F_expanded_nodes = 0
F_lg_expanded_nodes = 0
B_expanded_nodes = 0
A_expanded_nodes = 0
B_lg_expanded_nodes = 0

F_time = 0.00
F_lg_time = 0.00
B_time = 0.00
A_time = 0.00
B_lg_time = 0.00

# Main program
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # If user clicked close, close the screen
            done = True
        elif event.type == pygame.KEYDOWN:  # in the keydown library, pygame takes keyboard keys as input and response
            # accordingly
            if event.key == pygame.K_c:  # if we press c(clear) the screen will be cleared
                for row in range(ROWS):
                    for column in range(ROWS):
                        color = WHITE
                        if grid[row][column] != 1:
                            if grid[row][column] != 10:
                                if grid[row][column] != 5:
                                    color = WHITE
                                    grid[row][column] = 0
                                    pygame.draw.rect(screen, color, [(MARGIN + WIDTH) * column + MARGIN,
                                                                     (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
                                else:
                                    pygame.draw.rect(screen, GREEN, [(MARGIN + WIDTH) * column + MARGIN,
                                                                     (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
                            else:
                                pygame.draw.rect(screen, RED, [(MARGIN + WIDTH) * column + MARGIN,
                                                                 (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])

                pygame.display.flip()
                time = 0
                distance = 0
                print('Path Cleared')

            elif event.key == pygame.K_1:  # If we press 1, Repeated Forward A* will be called
                start_time = timeit.default_timer()
                result, F_expanded_nodes = repeated_a_star(grid, grid, start, target, ROWS, screen)
                stop_time = timeit.default_timer()
                pygame.display.flip()
                F_time = (stop_time - start_time)
                print(
                    'Running Repeated Forward A*: \nTime Taken: {} seconds \nDistance: {}'.format(F_time, F_expanded_nodes))
            elif event.key == pygame.K_2:  # If we press 2, Repeated Backward A* will be called
                start_time = timeit.default_timer()
                result, B_expanded_nodes = repeated_a_star(grid, grid, target, start, ROWS, screen)
                stop_time = timeit.default_timer()
                pygame.display.flip()
                B_time = (stop_time - start_time)
                print('Running Repeated Backwards A*: \nTime Taken: {} seconds \nDistance: {}'.format(B_time,
                                                                                                      B_expanded_nodes))
            elif event.key == pygame.K_3:  # If we press 3, Repeated Adaptive A* will be called
                start_time = timeit.default_timer()
                result, A_expanded_nodes = repeated_adaptive_a_star(grid, grid, start, target, ROWS, screen)
                stop_time = timeit.default_timer()
                pygame.display.flip()
                A_time = (stop_time - start_time)
                print('Running Adaptive A*: \nTime Taken: {} seconds \nDistance: {}'.format(A_time, A_expanded_nodes))
            elif event.key == pygame.K_4:  # If we press 1, Repeated Forward A* (large g) will be called
                start_time = timeit.default_timer()
                result, F_lg_expanded_nodes = repeated_a_star_with_large_g(grid, grid, start, target, ROWS, screen)
                stop_time = timeit.default_timer()
                pygame.display.flip()
                F_lg_time = (stop_time - start_time)
                print(
                    'Running Repeated Forward A* with large g: \nTime Taken: {} seconds \nDistance: {}'.format(F_lg_time, F_lg_expanded_nodes))
            elif event.key == pygame.K_5:  # If we press 1, Repeated Backward A* (large g) will be called
                start_time = timeit.default_timer()
                result, B_lg_expanded_nodes = repeated_a_star_with_large_g(grid, grid, target, start, ROWS, screen)
                stop_time = timeit.default_timer()
                pygame.display.flip()
                B_lg_time = (stop_time - start_time)
                print(
                    'Running Repeated Backward A* with large g: \nTime Taken: {} seconds \nDistance: {}'.format(B_lg_time, B_lg_expanded_nodes))

    pygame.display.flip()

pygame.quit()

# Create Bar Charts for comparing all the algorithms
time = [F_time, B_time, A_time, F_lg_time, B_lg_time]
distance = [F_expanded_nodes, B_expanded_nodes, A_expanded_nodes, F_lg_expanded_nodes, B_lg_expanded_nodes]
# Data
grid_worlds = ['Forward A*', 'Backward A*', 'Adaptive A*', 'Forward A* (Large g)', 'Backward A* (Large g)']

# Plot
fig, (ax1, ax2) = plt.subplots(2)

# Time subplot
width = 0.4  # Width of the bars
for i in range(len(grid_worlds)):
    ax1.bar(i, time[i], width=width, label='Time', color='b', align='center')

ax1.set_xlabel('Algorithms')
ax1.set_ylabel('Time (s)')
ax1.set_xticks(range(len(grid_worlds)))
ax1.set_xticklabels(grid_worlds)

# Distance subplot
for i in range(len(grid_worlds)):
    ax2.bar(i, distance[i], width=width, label='Distance', color='g', align='center')

ax2.set_xlabel('Algorithms')
ax2.set_ylabel('Distance')
ax2.set_xticks(range(len(grid_worlds)))
ax2.set_xticklabels(grid_worlds)

plt.suptitle(f"Gridworld {num} Information")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
