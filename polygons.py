from collections import namedtuple
from copy import deepcopy
from typing import Any, List, Set, Tuple
from math import inf


class PolygonsError(Exception):

    def __init__(self, message):
        super().__init__(message)

    @staticmethod
    def input_error():
        error_message = PolygonsError.construct_error_message('input')
        return PolygonsError(error_message)

    @staticmethod
    def no_expected_polygon():
        error_message = PolygonsError.construct_error_message('polygon')
        return PolygonsError(error_message)

    @staticmethod
    def construct_error_message(error_type):

        if error_type == 'input':
            return 'Incorrect input.'
        elif error_type == 'polygon':
            return 'Cannot get polygons as expected.'
        else:
            return 'Unknown error.'


class CoordinatePair(namedtuple('Coordinate', ['y', 'x'])):
    def __mul__(self, other) -> int:

        y_mult = self.y * other.x
        x_mult = self.x * other.y
        return y_mult - x_mult

    def __add__(self, other):

        new_y = self.y + other.y
        new_x = self.x + other.x
        return CoordinatePair(new_y, new_x)

    def __sub__(self, other):

        new_y = self.y - other.y
        new_x = self.x - other.x
        return CoordinatePair(new_y, new_x)

    def theneardir(self):

        return [self + CoordinatePair(dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

    def thenear_dirctior(self):

        return [self + CoordinatePair(dy, dx) for dy, dx in
                [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]]

    def max(self, other):

        new_y = max(self.y, other.y)
        new_x = max(self.x, other.x)
        return CoordinatePair(new_y, new_x)

    def min(self, other):

        new_y = min(self.y, other.y)
        new_x = min(self.x, other.x)
        return CoordinatePair(new_y, new_x)

    def check_in_range(self, xdim: int, ydim: int):

        in_x_range = 0 <= self.x < xdim
        in_y_range = 0 <= self.y < ydim
        return in_x_range and in_y_range

    def calculate_manhattan_distance(self, other):

        y_diff = abs(self.y - other.y)
        x_diff = abs(self.x - other.x)
        return y_diff + x_diff

    def findslope(self):

        if self.x == 0:
            return inf
        else:
            return self.y / self.x

    def ture_parallel(self, other):

        self_slope = self.findslope()
        other_slope = other.findslope()
        return self_slope == other_slope


class Grid:
    Shape = namedtuple('Shape', ['ydim', 'xdim'])

    def __init__(self, grid: List[List[int]]) -> None:
        self._shape = self.Shape(self.calculate_dimension(0, grid), self.calculate_dimension(1, grid))
        self._grid = grid

    def calculate_dimension(self, axis, grid):
        if axis == 0:
            return len(grid)
        elif axis == 1 and grid:
            return len(grid[0])
        else:
            return 0

    def __iter__(self):

        for row in self._grid:
            for item in row:
                yield item

    def __getitem__(self, index: Tuple[int, int]):

        y, x = index
        if y >= self._shape.ydim or x >= self._shape.xdim:
            raise IndexError("Index out of range")
        return self._grid[y][x]

    def __setitem__(self, index: Tuple[int, int], obj: int):

        y, x = index
        if y >= self._shape.ydim or x >= self._shape.xdim:
            raise IndexError("Index out of range")
        self._grid[y][x] = obj

    def likecopy(self):

        copied_grid = [[item for item in row] for row in self._grid]
        return Grid(copied_grid)

    def finalshape(self):
        return self._shape

    @staticmethod
    def all_zeros(shape: Tuple[int, int]):
        ydim, xdim = shape
        return Grid([[0 for _ in range(xdim)] for _ in range(ydim)])


class the_array_vertex(list):
    @staticmethod
    def roll(count: int, lst: List[Any]):

        if len(lst) == 0:
            return lst
        count %= len(lst)

        tail = lst[-count:] if count else []
        head = lst[:-count] if count else lst.copy()

        new_lst = []
        for item in tail:
            new_lst.append(item)
        for item in head:
            new_lst.append(item)

        return new_lst


class Polygon:
    origin: CoordinatePair
    depth: int
    all_points: List[CoordinatePair]
    rotations: int
    xdim: int
    new_found_points: Grid
    vertex: List[CoordinatePair]
    nb_vertex: int
    perimeter: str
    ydim: int
    convex: str
    dead_loop: Set[CoordinatePair]
    area: float
    INTERVAL = 0.4

    def __init__(self, coord: CoordinatePair, found_points: Grid, depth_map: Grid):
        self.ydim, self.xdim = depth_map.finalshape()
        self.depth_map = depth_map.likecopy()
        self.origin = coord
        self.new_found_points = found_points.likecopy()
        self.dead_loop = set()
        self.depth = depth_map[coord]
        self.all_points = self.the_boundaies([], coord)
        if not self.all_points:
            raise PolygonsError.no_expected_polygon()
        self.vertex = self.the_qulificated_vertex()

        for p in self.all_points:
            self.new_found_points[p] = 1
            self.depth_map[p] = self.depth
        self.nb_vertex = len(self.vertex)
        self.perimeter = self.all_the_perimeter()
        self.area = self.the_final_area()
        self.rotations = self.the_rotatations()
        self.convex = self.__the_finall_con()

    def _copy_depth_map(self, depth_map):
        return depth_map.likecopy()

    def _copy_found_points(self, found_points):
        return found_points.likecopy()

    def _set_found_points_and_depth(self):
        for p in self.all_points:
            self.new_found_points[p] = 1
            self.depth_map[p] = self.depth

    def the_qulificated_vertex(self) -> List[CoordinatePair]:
        v = [self.origin]
        idx = 1
        while idx < len(self.all_points):
            p = self.all_points[idx]
            last_point = v[-1]
            if idx > 1:
                pre_last_point = v[-2]
                parallel = (p - pre_last_point).ture_parallel(p - last_point)
                if parallel:
                    v[-1] = p
                else:
                    v.append(p)
            else:
                v.append(p)
            idx += 1

        if len(v) > 2:
            last = v[-1]
            second_last = v[-2]
            first = v[0]
            if (last - second_last).ture_parallel(first - second_last):
                v = v[:-1]

        for i in range(len(v)):
            if v[i] == self.origin:
                continue

        return v

    def the_qulificated_clocks(self, coord: CoordinatePair, from_coord: CoordinatePair) -> List[CoordinatePair]:
        next_list = coord.thenear_dirctior()
        roll_index = next_list.index(from_coord) + 1
        rolled_list = the_array_vertex.roll(-roll_index, next_list)
        truncated_list = rolled_list[:7]

        result_list = []
        for c in truncated_list:
            if not c.check_in_range(self.xdim, self.ydim):
                continue
            if self.depth_map[c] != self.depth:
                continue
            if c in self.dead_loop or self.new_found_points[c]:
                continue
            result_list.append(c)

        return result_list

    def the_boundaies(self, points: List[CoordinatePair], p: CoordinatePair) -> List[CoordinatePair]:
        if not points:
            if p == self.origin:
                prev = p + CoordinatePair(-1, 1)
            else:
                prev = None
        else:
            if p in points:
                if p == self.origin:
                    return points
                else:
                    return []
            else:
                prev = points[-1]

        new_points = points.copy()
        new_points.append(p)

        for the_down_coord in self.the_qulificated_clocks(p, prev):
            if the_down_coord in self.dead_loop:
                continue
            result = self.the_boundaies(new_points, the_down_coord)
            if result:
                if len(result) > 2:
                    return result
                else:
                    continue

        if len(new_points) > 3:
            self.dead_loop.add(p)
        return []

    def all_the_perimeter(self):
        perimeter_counts = [0, 0]
        for i, current_point in enumerate(self.all_points):
            previous_point = self.all_points[i - 1]
            distance = current_point.calculate_manhattan_distance(previous_point)
            perimeter_counts[distance - 1] += 1

        perimeter_parts = []
        if perimeter_counts[0] > 0:
            length_part = f'{self.INTERVAL * perimeter_counts[0]:.1f}'
            perimeter_parts.append(length_part)

        if perimeter_counts[1] > 0:
            sqrt_part = f'{perimeter_counts[1]}*sqrt(.32)'
            if perimeter_counts[0] > 0:
                perimeter_parts.append(' + ')
            perimeter_parts.append(sqrt_part)

        return ''.join(perimeter_parts)

    def the_final_area(self):
        y = the_array_vertex([vertex[0] for vertex in self.vertex])
        x = the_array_vertex([vertex[1] for vertex in self.vertex])

        y1 = the_array_vertex([y[(i + 1) % len(y)] for i in range(len(y))])
        x1 = the_array_vertex([x[(i + 1) % len(x)] for i in range(len(x))])

        area_parts = []
        for i in range(self.nb_vertex):
            part_area = y[i] * x1[i] - x[i] * y1[i]
            area_parts.append(part_area)

        total_area = sum(area_parts)
        abs_area = abs(total_area)

        area_result = abs_area / 2
        scaled_area = area_result * self.INTERVAL ** 2
        return scaled_area

    def __the_finall_con(self) -> str:
        convexity_checks = []
        for i, current_vertex in enumerate(self.vertex):
            prev_vertex = self.vertex[i - 1]
            next_vertex = self.vertex[(i + 1) % self.nb_vertex]

            cross_product = (next_vertex - prev_vertex) * (current_vertex - prev_vertex)
            convexity_checks.append(cross_product >= 0)

        if all(convexity_checks):
            return 'yes'
        else:

            for index, is_convex in enumerate(convexity_checks):
                if not is_convex:
                    break
            return 'no'

    def the_rotatations(self):
        furthest = CoordinatePair(0, 0)
        nearest = CoordinatePair(199, 199)
        for c in self.vertex:
            furthest = c.max(furthest)
            nearest = c.min(nearest)
        (height, width) = furthest - nearest
        vertex = [furthest - c for c in self.vertex]
        vertex_4 = [CoordinatePair(x, -y) for y, x in vertex]

        vertex_2 = [CoordinatePair(-c.y, -c.x) for c in vertex]

        rotations = 0

        for c in vertex:
            if (c.y, c.x - width) not in vertex_4:
                break
        else:
            return 4

        for c in vertex:
            if CoordinatePair(c.y - height, c.x - width) not in vertex_2:
                break
        else:
            return 2

        return 1

    def __str__(self):
        parts = ['    Perimeter: ']
        if isinstance(self.perimeter, str) and self.perimeter:
            parts.append(self.perimeter)
        else:
            parts.append('Unknown')

        parts.append('\n    Area: ')
        if isinstance(self.area, (int, float)):
            parts.append(f'{self.area:.2f}')
        else:
            parts.append('Unknown')

        parts.append('\n    Convex: ')
        parts.append(self.convex if self.convex in ['yes', 'no'] else 'Unknown')

        parts.append('\n    Nb of invariant rotations: ')
        if isinstance(self.rotations, int):
            parts.append(str(self.rotations))
        else:
            parts.append('Unknown')

        parts.append('\n    Depth: ')
        parts.append(str(self.depth) if isinstance(self.depth, int) else 'Unknown')

        return ''.join(parts)


class Polygons:
    plist: List[Polygon]
    xdim: int
    ydim: int

    def __init__(self, polys_input_file: str):

        self.name = polys_input_file.rstrip('txt')

        with open(polys_input_file) as file:
            lines = file.readlines()

        processed_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                no_space_line = stripped_line.replace(' ', '')
                processed_lines.append(no_space_line)

        self.ydim = len(processed_lines)
        self.xdim = 0 if not processed_lines else len(processed_lines[0])

        if not (2 <= self.ydim <= 50) or not (2 <= self.xdim <= 50):
            raise PolygonsError.input_error()

        for line in processed_lines:
            if len(line) != self.xdim or any(ch not in ['0', '1'] for ch in line):
                raise PolygonsError.input_error()

        def the_next_depth(coord: CoordinatePair, cur_depth):
            nears = []
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                potential_near = CoordinatePair(coord.y + direction[0], coord.x + direction[1])
                if potential_near.check_in_range(self.xdim, self.ydim):
                    nears.append(potential_near)

            boundary_check = []
            for near_coord in nears:
                if depth[near_coord] < cur_depth:
                    boundary_check.append(True)
                else:
                    boundary_check.append(False)

            if True in boundary_check or len(nears) != 4:
                return cur_depth
            else:
                for check in boundary_check:
                    if not check:
                        return 99
                return 99

        def bfs(searched: Grid, depth_now: int, coord: CoordinatePair):
            queue = [coord]
            visited = set()

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                if searched[current] or (depth[current] > depth_now and grid[current]):
                    visited.add(current)
                    continue

                if not grid[current]:
                    depth[current] = -1

                searched[current] = 1
                visited.add(current)

                for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    potential_near = CoordinatePair(current.y + direction[0], current.x + direction[1])
                    if potential_near.check_in_range(self.xdim, self.ydim):
                        queue.append(potential_near)

        def path_edge(depth_now: int):
            searched = Grid([[0 for _ in range(self.xdim)] for _ in range(self.ydim)])

            for x in range(self.xdim):
                for y in [0, self.ydim - 1]:
                    coord = CoordinatePair(y, x)
                    if coord not in searched:
                        bfs(searched, depth_now, coord)

            for y in range(self.ydim):
                for x in [0, self.xdim - 1]:
                    coord = CoordinatePair(y, x)
                    if coord not in searched:
                        bfs(searched, depth_now, coord)

        self.plist = []
        grid = Grid([[int(char) for char in row] for row in processed_lines])
        depth = Grid([[99 for _ in range(self.xdim)] for _ in range(self.ydim)])

        current_depth = -1
        finished = False
        while not finished:
            path_edge(current_depth)
            current_depth += 1
            finished = True

            for y in range(self.ydim):
                for x in range(self.xdim):
                    if depth[y, x] == 99 and grid[y, x]:
                        finished = False
                        the_next_depth_val = the_next_depth(CoordinatePair(y, x), current_depth)
                        depth[y, x] = the_next_depth_val

        self.max_depth = current_depth - 1

        found = Grid.all_zeros(grid.finalshape())
        for y in range(self.ydim):
            for x in range(self.xdim):
                coord = CoordinatePair(y, x)
                if self.should_add_polygon(found, depth, coord):
                    polygon = Polygon(coord, found, depth)
                    self.plist.append(polygon)
                    found = polygon.new_found_points

        if self.any_unfound_polygon(grid, found):
            raise PolygonsError.no_expected_polygon()

    def should_add_polygon(self, found, depth, coord):
        return not found[coord.y, coord.x] and depth[coord.y, coord.x] > -1

    def any_unfound_polygon(self, grid, found):
        for y in range(grid.finalshape().ydim):
            for x in range(grid.finalshape().xdim):
                if not found[y, x] and grid[y, x]:
                    return True
        return False

    def analyse(self):
        polygon_count = len(self.plist)
        for i in range(polygon_count):
            p = self.plist[i]
            print(f'Polygon {i + 1}:')
            print(p)

    def display(self):
        def outline():
            outline_commands = []
            outline_commands.append(f'\\draw[ultra thick] (0, 0) -- ({self.xdim - 1}, 0)')
            outline_commands.append(f' -- ({self.xdim - 1}, {self.ydim - 1})')
            outline_commands.append(f' -- (0, {self.ydim - 1}) -- cycle;')
            return ''.join(outline_commands)

        def polys(gap):
            polys_commands = []
            for depth in range(self.max_depth + 1):
                polys_commands.append(f'\n% Depth {depth}')
                for poly in self.plist:
                    if poly.depth != depth:
                        continue
                    color = 0 if not gap else round((max_area - poly.area) / gap * 100)
                    poly_command = f'\n\\filldraw[fill=orange!{color}!yellow] '
                    for point in poly.vertex:
                        poly_command += f'({point[1]}, {point[0]}) -- '
                    poly_command += 'cycle;'
                    polys_commands.append(poly_command)
            return ''.join(polys_commands)

        areas = [p.area for p in self.plist if self.plist]
        max_area = max(areas) if areas else 0
        min_area = min(areas) if areas else 0
        ranging = max_area - min_area

        with open(self.name + 'tex', 'w') as file:
            file_contents = f'''\\documentclass[10pt]{{article}}
\\usepackage{{tikz}}
\\usepackage[margin=0cm]{{geometry}}
\\pagestyle{{empty}}

\\begin{{document}}

\\vspace*{{\\fill}}
\\begin{{center}}
\\begin{{tikzpicture}}[x=0.4cm, y=-0.4cm, thick, brown]
{outline()}
{polys(ranging)}
\\end{{tikzpicture}}
\\end{{center}}
\\vspace*{{\\fill}}

\\end{{document}}
'''
            file.write(file_contents)
