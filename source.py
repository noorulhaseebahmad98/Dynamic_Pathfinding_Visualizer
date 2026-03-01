import pygame
import sys
import heapq
import time
import random
import math

# ---------------------------- Constants & Theme ----------------------------
CELL_SIZE = 30
PANEL_WIDTH = 340
MAX_GRID_SIZE = 50

# Elegant Brown & Classic Golden Palette
BG_COLOR = (44, 26, 18)
PANEL_COLOR = (61, 38, 26)
GRID_BG = (84, 56, 40)
OBSTACLE_COLOR = (26, 15, 10)

# Visual Highlights
FRONTIER_COLOR = (241, 196, 15)     # Yellow (Priority Queue)
VISITED_COLOR = (52, 152, 219)      # Blue (Explored Nodes)
PATH_COLOR = (46, 204, 113)         # Green (Optimal Path)
CURRENT_COLOR = (230, 126, 34)      # Orange (Currently evaluating node)

START_COLOR = (255, 255, 255)       # White
GOAL_COLOR = (231, 76, 60)          # Red
AGENT_COLOR = (212, 175, 55)        # Classic Gold

BUTTON_COLOR = (84, 56, 40)
BUTTON_HOVER = (110, 75, 53)
BUTTON_ACTIVE = (212, 175, 55)
TEXT_COLOR = (245, 222, 179)
TEXT_DARK = (44, 26, 18)

# ---------------------------- Grid Class ----------------------------
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = (0, 0)
        self.goal = (rows-1, cols-1)

    def set_obstacle(self, row, col, value):
        if (row, col) not in (self.start, self.goal):
            self.cells[row][col] = value

    def is_obstacle(self, row, col):
        return self.cells[row][col] == 1

    def toggle_obstacle(self, row, col):
        if (row, col) not in (self.start, self.goal):
            self.cells[row][col] = 1 - self.cells[row][col]

    def random_generate(self, density):
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.goal):
                    continue
                self.cells[r][c] = 1 if random.random() < density else 0

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and not self.is_obstacle(row, col)

# ---------------------------- Search Algorithms ----------------------------
def heuristic_manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def heuristic_euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Generator function for step-by-step execution
def search_generator(grid, start, goal, algorithm, heuristic_name):
    heuristic = heuristic_manhattan if heuristic_name == "Manhattan" else heuristic_euclidean
    open_set = []
    closed_set = set()
    g_values = {start: 0}
    parent = {start: None}
    
    h_start = heuristic(start, goal)
    f_start = h_start if algorithm == "GBFS" else g_values[start] + h_start
    heapq.heappush(open_set, (f_start, start))
    frontier_set = {start}
    
    nodes_visited = 0
    start_time = time.time()

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
            
        frontier_set.discard(current)
        closed_set.add(current)
        nodes_visited += 1

        # Yield current state BEFORE checking neighbors so UI can update
        yield None, closed_set, frontier_set, nodes_visited, 0, (time.time() - start_time) * 1000, current

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            exec_time = (time.time() - start_time) * 1000
            yield path, closed_set, frontier_set, nodes_visited, len(path)-1, exec_time, current
            return

        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = current[0]+dr, current[1]+dc
            neighbor = (nr, nc)
            
            if grid.is_valid(nr, nc):
                tentative_g = g_values[current] + 1
                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    parent[neighbor] = current
                    h = heuristic(neighbor, goal)
                    f = h if algorithm == "GBFS" else tentative_g + h
                    
                    if neighbor not in closed_set:
                        heapq.heappush(open_set, (f, neighbor))
                        frontier_set.add(neighbor)

    exec_time = (time.time() - start_time) * 1000
    yield None, closed_set, frontier_set, nodes_visited, 0, exec_time, None

# ---------------------------- UI Components ----------------------------
def draw_rounded_rect(surface, color, rect, radius=6):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hovered = False
        self.active = False
        self.disabled = False

    def handle_event(self, event):
        if self.disabled: return
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.callback()

    def draw(self, screen, font):
        if self.disabled:
            color = (50, 40, 35)
            txt_color = (100, 80, 70)
        else:
            color = BUTTON_ACTIVE if self.active else (BUTTON_HOVER if self.hovered else BUTTON_COLOR)
            txt_color = TEXT_DARK if self.active else TEXT_COLOR
        
        draw_rounded_rect(screen, (26, 15, 10), self.rect.move(2, 3))
        draw_rounded_rect(screen, color, self.rect)
        
        text_surf = font.render(self.text, True, txt_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.active = False
        self.cursor_visible = True
        self.last_blink = time.time()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit() and len(self.text) < 3:
                self.text += event.unicode

    def draw(self, screen, font):
        draw_rounded_rect(screen, OBSTACLE_COLOR, self.rect.move(2, 2))
        bg_color = BUTTON_ACTIVE if self.active else BUTTON_COLOR
        draw_rounded_rect(screen, bg_color, self.rect)
        
        txt_color = TEXT_DARK if self.active else TEXT_COLOR
        txt_surface = font.render(self.text, True, txt_color)
        
        text_x = self.rect.x + (self.rect.width - txt_surface.get_width()) // 2
        text_y = self.rect.y + (self.rect.height - txt_surface.get_height()) // 2
        screen.blit(txt_surface, (text_x, text_y))
        
        if self.active:
            if time.time() - self.last_blink > 0.5:
                self.cursor_visible = not self.cursor_visible
                self.last_blink = time.time()
            if self.cursor_visible:
                cursor_x = text_x + txt_surface.get_width() + 2
                pygame.draw.line(screen, txt_color, (cursor_x, text_y + 4), (cursor_x, text_y + txt_surface.get_height() - 4), 2)

# ---------------------------- Main Application ----------------------------
class App:
    def __init__(self):
        pygame.init()
        self.rows, self.cols = 20, 25
        self.grid = Grid(self.rows, self.cols)
        
        self.setup_display()
        self.clock = pygame.time.Clock()
        
        font_names = "segoeui, arial, sans-serif"
        self.font = pygame.font.SysFont(font_names, 16, bold=True)
        self.small_font = pygame.font.SysFont(font_names, 14)
        self.title_font = pygame.font.SysFont(font_names, 22, bold=True)

        # Search Data
        self.algorithm = "A*"
        self.heuristic = "Manhattan"
        self.path = None
        self.visited = set()
        self.frontier = set()
        self.current_node = None
        self.search_gen = None  # Holds the step-by-step generator
        
        self.metrics = {"nodes": 0, "cost": 0, "time": 0.0, "status": "Idle"}
        
        # Dynamic Mode
        self.dynamic_mode = False
        self.agent_pos = None
        self.agent_path = []
        self.last_update = 0
        self.update_interval = 250
        self.obstacle_chance = 0.15

        self.buttons = []
        self.input_boxes = []
        self.create_ui()

    def setup_display(self):
        self.screen_width = self.cols * CELL_SIZE + PANEL_WIDTH
        self.screen_height = max(self.rows * CELL_SIZE, 720)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pathfinding Visualizer Pro")

    def create_ui(self):
        self.buttons.clear()
        self.input_boxes.clear()
        
        x = self.cols * CELL_SIZE + 20
        y = 20
        w = PANEL_WIDTH - 40
        h = 32
        spacing = 10

        self.rows_box = InputBox(x + 50, y, 60, h, str(self.rows))
        self.cols_box = InputBox(x + 170, y, 60, h, str(self.cols))
        self.input_boxes.extend([self.rows_box, self.cols_box])
        
        y += h + spacing
        self.buttons.append(Button(x, y, w, h, "Apply New Size", self.set_size))

        y += h + spacing + 10
        self.density = 0.3
        self.buttons.append(Button(x, y, 40, h, "-", self.dec_density))
        self.buttons.append(Button(x + w - 40, y, 40, h, "+", self.inc_density))
        self.buttons.append(Button(x, y + h + spacing, w, h, "Generate Maze", self.random_gen))

        y += h * 2 + spacing * 2 + 10
        self.alg_a = Button(x, y, w//2 - 5, h, "A*", lambda: self.set_alg("A*"))
        self.alg_g = Button(x + w//2 + 5, y, w//2 - 5, h, "GBFS", lambda: self.set_alg("GBFS"))
        self.buttons.extend([self.alg_a, self.alg_g])
        
        y += h + spacing
        self.heur_m = Button(x, y, w//2 - 5, h, "Manhattan", lambda: self.set_heur("Manhattan"))
        self.heur_e = Button(x + w//2 + 5, y, w//2 - 5, h, "Euclidean", lambda: self.set_heur("Euclidean"))
        self.buttons.extend([self.heur_m, self.heur_e])

        # Core Controls
        y += h + spacing + 10
        self.btn_run_inst = Button(x, y, w//2 - 5, h, "Run Instant", self.run_search_instant)
        self.btn_run_step = Button(x + w//2 + 5, y, w//2 - 5, h, "Start Stepping", self.start_step_by_step)
        self.buttons.extend([self.btn_run_inst, self.btn_run_step])

        y += h + spacing
        self.btn_next_step = Button(x, y, w, h, "Next Step (Spacebar)", self.do_next_step)
        self.btn_next_step.disabled = True
        self.buttons.append(self.btn_next_step)
        
        y += h + spacing + 10
        self.buttons.append(Button(x, y, w//2 - 5, h, "Start Dynamic", self.start_dynamic))
        self.buttons.append(Button(x + w//2 + 5, y, w//2 - 5, h, "Stop Dynamic", self.stop_dynamic))

        self.metrics_y = y + h + spacing + 15
        self.update_active_buttons()

    def update_active_buttons(self):
        self.alg_a.active = (self.algorithm == "A*")
        self.alg_g.active = (self.algorithm == "GBFS")
        self.heur_m.active = (self.heuristic == "Manhattan")
        self.heur_e.active = (self.heuristic == "Euclidean")
        
        # Toggle Next Step availability
        if hasattr(self, 'btn_next_step'):
            self.btn_next_step.disabled = (self.search_gen is None)

    def set_size(self):
        try:
            r = min(int(self.rows_box.text) if self.rows_box.text else self.rows, MAX_GRID_SIZE)
            c = min(int(self.cols_box.text) if self.cols_box.text else self.cols, MAX_GRID_SIZE)
            if r > 2 and c > 2:
                self.rows, self.cols = r, c
                self.grid = Grid(r, c)
                self.setup_display()
                self.create_ui()
                self.reset_search_state()
        except ValueError:
            pass

    def dec_density(self): self.density = max(0.0, self.density - 0.1)
    def inc_density(self): self.density = min(1.0, self.density + 0.1)

    def random_gen(self):
        self.grid.random_generate(self.density)
        self.reset_search_state()

    def set_alg(self, alg):
        self.algorithm = alg
        self.update_active_buttons()

    def set_heur(self, heur):
        self.heuristic = heur
        self.update_active_buttons()

    def reset_search_state(self):
        self.path = None
        self.visited.clear()
        self.frontier.clear()
        self.current_node = None
        self.search_gen = None
        self.metrics = {"nodes": 0, "cost": 0, "time": 0.0, "status": "Idle"}
        self.update_active_buttons()

    def run_search_instant(self):
        self.reset_search_state()
        # Consume the generator fully
        gen = search_generator(self.grid, self.grid.start, self.grid.goal, self.algorithm, self.heuristic)
        final_state = None
        for state in gen:
            final_state = state
            
        if final_state and final_state[0]:
            self.path, self.visited, self.frontier, nodes, cost, t, _ = final_state
            self.metrics = {"nodes": nodes, "cost": cost, "time": t, "status": "Path Found!"}
        else:
            self.visited, self.frontier = final_state[1], final_state[2]
            self.metrics = {"nodes": final_state[3], "cost": 0, "time": final_state[5], "status": "No Path"}

    def start_step_by_step(self):
        self.reset_search_state()
        self.search_gen = search_generator(self.grid, self.grid.start, self.grid.goal, self.algorithm, self.heuristic)
        self.metrics["status"] = "Stepping..."
        self.update_active_buttons()
        self.do_next_step() # do first step automatically

    def do_next_step(self):
        if self.search_gen is None:
            return
        
        try:
            state = next(self.search_gen)
            self.path, self.visited, self.frontier, nodes, cost, t, self.current_node = state
            self.metrics.update({"nodes": nodes, "cost": cost, "time": t})
            
            # If path is found or generator completes
            if self.path is not None:
                self.metrics["status"] = "Path Found!"
                self.search_gen = None
                self.current_node = None
        except StopIteration:
            self.metrics["status"] = "No Path Found"
            self.search_gen = None
            self.current_node = None
            
        self.update_active_buttons()

    def start_dynamic(self):
        self.run_search_instant()
        if self.path:
            self.dynamic_mode = True
            self.agent_pos = self.grid.start
            self.agent_path = self.path[1:]
            self.last_update = pygame.time.get_ticks()

    def stop_dynamic(self):
        self.dynamic_mode = False
        self.agent_pos = None
        self.agent_path = []
        self.metrics["status"] = "Dynamic Stopped"

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.search_gen is not None:
                    self.do_next_step()
                
            if not self.dynamic_mode and event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < self.cols * CELL_SIZE:
                    row, col = y // CELL_SIZE, x // CELL_SIZE
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        if event.button == 1:
                            self.grid.toggle_obstacle(row, col)
                        elif event.button == 3:
                            if not self.grid.is_obstacle(row, col): self.grid.start = (row, col)
                        elif event.button == 2:
                            if not self.grid.is_obstacle(row, col): self.grid.goal = (row, col)
                        self.reset_search_state()

            for btn in self.buttons: btn.handle_event(event)
            for box in self.input_boxes: box.handle_event(event)

    def update_dynamic(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.update_interval:
            self.last_update = now
            
            if self.agent_pos == self.grid.goal:
                self.stop_dynamic()
                self.metrics["status"] = "Goal Reached!"
                return

            path_blocked = any(self.grid.is_obstacle(*pos) for pos in self.agent_path)
            if path_blocked or not self.agent_path:
                self.replan()

            if self.agent_path:
                self.agent_pos = self.agent_path.pop(0)
                self.metrics["status"] = "Moving..."
            else:
                self.metrics["status"] = "Blocked / Unreachable"

            if random.random() < self.obstacle_chance:
                self.spawn_obstacle()

    def spawn_obstacle(self):
        empty_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                       if not self.grid.is_obstacle(r, c) and (r, c) not in (self.grid.start, self.grid.goal, self.agent_pos)]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid.set_obstacle(r, c, 1)

    def replan(self):
        if not self.agent_pos: return
        self.reset_search_state()
        gen = search_generator(self.grid, self.agent_pos, self.grid.goal, self.algorithm, self.heuristic)
        final_state = None
        for state in gen: final_state = state
        
        if final_state and final_state[0]:
            self.path, self.visited, self.frontier, nodes, cost, t, _ = final_state
            self.agent_path = self.path[1:]
            self.metrics.update({"nodes": nodes, "cost": cost, "time": t})
        else:
            self.agent_path = []
            self.path = None

    def draw_grid(self):
        pygame.draw.rect(self.screen, BG_COLOR, (0, 0, self.cols * CELL_SIZE, self.rows * CELL_SIZE))

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * CELL_SIZE + 1, r * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                
                if self.grid.is_obstacle(r, c):
                    pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect, border_radius=4)
                elif self.path and (r, c) in self.path and (r, c) not in (self.grid.start, self.grid.goal):
                    pygame.draw.rect(self.screen, PATH_COLOR, rect, border_radius=4)
                elif self.current_node and (r, c) == self.current_node and (r, c) not in (self.grid.start, self.grid.goal):
                    pygame.draw.rect(self.screen, CURRENT_COLOR, rect, border_radius=4)
                elif (r, c) in self.visited and (r, c) not in (self.grid.start, self.grid.goal):
                    pygame.draw.rect(self.screen, VISITED_COLOR, rect, border_radius=4)
                elif (r, c) in self.frontier and (r, c) not in (self.grid.start, self.grid.goal):
                    pygame.draw.rect(self.screen, FRONTIER_COLOR, rect, border_radius=4)
                else:
                    if (r, c) not in (self.grid.start, self.grid.goal):
                        pygame.draw.rect(self.screen, GRID_BG, rect, border_radius=4)

        def draw_node(pos, color, radius_ratio):
            center = (pos[1] * CELL_SIZE + CELL_SIZE//2, pos[0] * CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(self.screen, color, center, int(CELL_SIZE * radius_ratio))

        draw_node(self.grid.start, START_COLOR, 0.4)
        draw_node(self.grid.goal, GOAL_COLOR, 0.4)

        if self.dynamic_mode and self.agent_pos:
            draw_node(self.agent_pos, AGENT_COLOR, 0.45)

    def draw_panel(self):
        x = self.cols * CELL_SIZE
        panel_rect = pygame.Rect(x, 0, PANEL_WIDTH, self.screen_height)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)
        pygame.draw.line(self.screen, BUTTON_ACTIVE, (x, 0), (x, self.screen_height), 2)

        for btn in self.buttons: btn.draw(self.screen, self.font)
        for box in self.input_boxes: box.draw(self.screen, self.font)

        self.screen.blit(self.small_font.render("Rows:", True, TEXT_COLOR), (x + 10, 28))
        self.screen.blit(self.small_font.render("Cols:", True, TEXT_COLOR), (x + 130, 28))
        
        density_txt = self.font.render(f"Density: {int(self.density*100)}%", True, TEXT_COLOR)
        self.screen.blit(density_txt, (x + PANEL_WIDTH//2 - density_txt.get_width()//2, 105))

        # Real-Time Metrics Dashboard
        metrics_rect = pygame.Rect(x + 20, self.metrics_y, PANEL_WIDTH - 40, 150)
        draw_rounded_rect(self.screen, BUTTON_COLOR, metrics_rect, 8)
        pygame.draw.rect(self.screen, BUTTON_ACTIVE, metrics_rect, 2, border_radius=8)
        
        self.screen.blit(self.title_font.render("Real-Time Metrics", True, BUTTON_ACTIVE), (x + 35, self.metrics_y + 10))
        
        stats = [
            f"Status: {self.metrics['status']}",
            f"Frontier Nodes: {len(self.frontier)}",
            f"Nodes Visited: {self.metrics['nodes']}",
            f"Path Cost: {self.metrics['cost']}"
        ]
        
        for i, stat in enumerate(stats):
            color = TEXT_COLOR
            if i == 0 and ("Found" in stat or "Reached" in stat): color = PATH_COLOR
            if i == 0 and "Block" in stat: color = GOAL_COLOR
            surf = self.small_font.render(stat, True, color)
            self.screen.blit(surf, (x + 35, self.metrics_y + 45 + (i * 22)))

        # Color Legend
        inst_y = self.screen_height - 120
        instructions = [
            "L-Click: Wall | R-Click: Start | M-Click: Goal",
            "Yellow: Frontier | Blue: Visited",
            "Orange: Current | Green: Path"
        ]
        for i, text in enumerate(instructions):
            self.screen.blit(self.small_font.render(text, True, (190, 170, 150)), (x + 20, inst_y + (i * 22)))

    def run(self):
        while True:
            self.handle_events()
            if self.dynamic_mode:
                self.update_dynamic()
                
            self.screen.fill(BG_COLOR)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    app = App()
    app.run()