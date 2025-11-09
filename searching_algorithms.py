from math import sqrt

from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False
    queue=deque()
    queue.append(start)
    visited={start}
    came_from={}
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current=queue.popleft()
        if current == end:
            while current in came_from:
                current=came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False
    stack=[start]
    visited={start}
    came_from={}
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current=stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    distance=abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
    return distance

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    distance=sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return distance


def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    count=0
    open_heap=PriorityQueue()
    open_heap.put((0, count, start))
    came_from={}
    g_score={spot:float('inf') for row in grid.grid for spot in row}
    g_score[start]=0
    f_score = {spot: float('inf') for row in grid.grid for spot in row}
    f_score[h_manhattan_distance(start.get_position(), end.get_position())] = 0
    lookup_set = {start}
    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current=open_heap.get()[2]
        lookup_set.remove(current)
        if current == end:
            while current in came_from:
                current=came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True
        for neighbor in current.neighbors:
            tentative_g=g_score[current]+1
            if tentative_g<g_score[neighbor]:
                came_from[neighbor]=current
                g_score[neighbor]=tentative_g
                f_score[neighbor]=tentative_g+h_manhattan_distance(neighbor.get_position(), end.get_position())
                if neighbor not in lookup_set:
                    count+=1
                    open_heap.put((f_score[neighbor], count, neighbor))
                    lookup_set.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
    if start is None or end is None:
        return False
    stack = [(start, 0)]
    visited = {start}
    came_from = {}
    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current, depth = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        if depth < limit:
            for neighbor in current.neighbors:
                if neighbor not in visited and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start))
    came_from = {}
    g_score = {spot: float('inf') for row in grid.grid for spot in row}
    g_score[start] = 0
    lookup_set = {start}

    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_heap.get()[2]
        lookup_set.remove(current)
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                if neighbor not in lookup_set:
                    count += 1
                    open_heap.put((g_score[neighbor], count, neighbor))
                    lookup_set.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def greedy(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start))
    came_from = {}
    visited = {start}

    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_heap.get()[2]
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                h = h_euclidian_distance(neighbor.get_position(), end.get_position())
                count += 1
                open_heap.put((h, count, neighbor))
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def iddfs(draw: callable, grid: Grid, start: Spot, end: Spot, max_depth: int) -> bool:
    if start is None or end is None:
        return False

    for depth in range(max_depth + 1):
        for row in grid.grid:
            for spot in row:
                if not spot.is_barrier() and spot != start and spot != end:
                    spot.reset()

        found = dls(draw, grid, start, end, depth)
        if found:
            print(f"Goal found at depth {depth}")
            return True

    print("Goal not found.")
    return False
def ida(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    def dfs_helper() -> tuple[dict, float]:
        stack = [start]
        costs = {start: 0}
        came_from = {}
        visited = set()
        pruned_bound = float('inf')

        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            draw()
            current = stack.pop()
            cost = costs[current]
            if current.get_position() == end.get_position():
                return came_from, 0

            if current.get_position() != start.get_position():
                current.make_closed()

            for neighbor in current.neighbors:
                new_cost = cost + 1
                new_bound = h_manhattan_distance(neighbor.get_position(), end.get_position()) + new_cost

                if neighbor not in visited:
                    if new_bound > bound:
                        if new_bound < pruned_bound:
                            pruned_bound = new_bound
                    else:
                        came_from[neighbor] = current
                        costs[neighbor] = new_cost
                        visited.add(neighbor)
                        stack.append(neighbor)
                        neighbor.make_open()

                elif costs[neighbor] > new_cost:
                    came_from[neighbor] = current
                    costs[neighbor] = new_cost
                    for i in range(len(stack)):
                        if stack[i].get_position() == neighbor.get_position():
                            break
                    else:
                        stack.append(neighbor)
                        neighbor.make_open()

        return None, pruned_bound

    bound = h_manhattan_distance(start.get_position(), end.get_position())
    found = False

    while not found:
        for row in grid.grid:
            for cell in row:
                if cell.is_closed() or cell.is_open():
                    cell.reset()

        came_from, new_bound = dfs_helper()
        draw()

        if came_from is not None:
            current = end
            while current.get_position() != start.get_position():
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        bound = new_bound
        print(bound)

    return False

# Assume that each edge (graph weight) equals