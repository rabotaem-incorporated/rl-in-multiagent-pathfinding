import pygame
import pygame.draw as draw
import pygame.locals
from pygame import Color, Rect, Vector2
import math
import abc
import pickle
import typing
import cv2
import tqdm
import dataclasses
from pathlib import Path
import numpy as np
import sys
import argparse


@dataclasses.dataclass(frozen=True)
class Run:
    map_: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    positions: np.ndarray
    attentions: np.ndarray
    num_agents: int
    map_width: int
    map_height: int


def load_run(map_path, result_path, run_id: int) -> Run:
    with open(map_path, "rb") as f:
        map_, starts, goals = pickle.load(f)[run_id]
        num_agents = len(starts)
        assert starts.shape == (num_agents, 2)
        assert goals.shape == (num_agents, 2)
        map_height, map_width = map_.shape
        assert np.all(map_ >= 0) and np.all(map_ <= 1)
        map_ = np.array(map_, dtype=np.bool_)
        starts = np.array(starts, dtype=np.int32)
        goals = np.array(goals, dtype=np.int32)
    with open(result_path, "rb") as f:
        data = pickle.load(f)
        positions = np.array(data[3][run_id], dtype=np.int32)
        attentions = np.array(data[4][run_id], dtype=np.float32)
        num_steps = len(positions)
        assert positions.shape == (num_steps, num_agents, 2)
        # assert attentions.shape == (num_steps, num_agents, num_agents)
        assert np.all(attentions >= 0) and np.all(attentions <= 1)
        assert np.all(positions >= 0) and np.all(positions < map_.shape)
        # assert np.all(positions[-1] == goals)
    positions = np.concatenate([starts[None], positions], axis=0)
    print(positions.shape)
    attentions = np.concatenate([np.zeros((1, *attentions.shape[1:])), attentions], axis=0)
    return Run(map_, starts, goals, positions, attentions, num_agents, map_width, map_height)


@dataclasses.dataclass
class Config:
    TILE_SIZE = 16
    BORDER_WIDTH = 1
    COMM_ARC_ANGLE = math.radians(30)
    FONT_NAME = None

    def smooth(self, t):
        t = np.clip(t, 0, 1)
        return 3 * t ** 2 - 2 * t ** 3

CONFIG = Config()

class RenderingContext:
    def __init__(self, screen: pygame.Surface, size: Vector2, dt: float):
        self.screen: pygame.Surface = screen
        self.offset: Vector2 = Vector2(0, 0)
        self.size: Vector2 = size
        self.dt: float = dt
        self.hitboxes: list[tuple[Rect, 'Widget']] = []

    def add_hitbox(self, rect: Rect, widget: 'Widget') -> None:
        self.hitboxes.append((rect, widget))

    def subrect_abs(self, rect: Rect) -> 'RenderingContext':
        rc = RenderingContext(self.screen, rect.size, self.dt)
        rc.offset = self.offset + rect.topleft
        rc.hitboxes = self.hitboxes
        return rc


class Widget(abc.ABC):
    def children(self) -> typing.Generator['Widget', None, None]:
        yield from []

    @abc.abstractmethod
    def min_size(self) -> Vector2:
        pass

    def on_click(self) -> None:
        pass

    def on_drag(self, delta: Vector2) -> None:
        pass

    def recompute_min_size(self) -> None:
        for child in self.children():
            child.recompute_min_size()
        self.minsize = self.min_size()

    @abc.abstractmethod
    def draw(self, rc: RenderingContext) -> None:
        ...


class RunWidget(Widget):
    def __init__(self, run: Run):
        self.run = run
        self.paused = True
        self.time = 0.0
        self.speed = 1.0

    def draw(self, rc: RenderingContext) -> None:
        if not self.paused:
            self.time += rc.dt * self.speed
        tile_size = Vector2(rc.size).elementwise() / Vector2(self.run.map_width, self.run.map_height)
        tile_size = int(min(tile_size.x, tile_size.y))
        for i in range(self.run.map_height):
            for j in range(self.run.map_width):
                draw.rect(
                    rc.screen,
                    '#999999' if self.run.map_[i, j] else '#ffffff',
                    Rect(
                        j * tile_size + CONFIG.BORDER_WIDTH,
                        i * tile_size + CONFIG.BORDER_WIDTH,
                        tile_size + 1 - 2 * CONFIG.BORDER_WIDTH,
                        tile_size + 1 - 2 * CONFIG.BORDER_WIDTH,
                    ).move(rc.offset + (rc.size - self.minsize) / 2))

        start = self.run.starts
        goal = self.run.goals
        max_time = len(self.run.positions) - 1
        pos1 = self.run.positions[min(int(self.time), max_time)]
        pos2 = self.run.positions[min(int(self.time) + 1, max_time)]
        attention1 = self.run.attentions[min(int(self.time), max_time)]
        attention2 = self.run.attentions[min(int(self.time) + 1, max_time)] 
        t = self.time % 1
        if t < 0.5:
            t = CONFIG.smooth(t * 2)
            pos = pos1
            attention = attention1 * (1 - t) + attention2 * t
        else:
            t = CONFIG.smooth((t - 0.5) * 2)
            pos = pos1 * (1 - t) + pos2 * t
            attention = attention2
        if len(attention.shape) == 3:
            attention = attention[0]
        attention = ((attention + attention.T) / 2) ** 0.4
        for i in range(self.run.num_agents):
            color = cv2.cvtColor(np.uint8([[[256 / self.run.num_agents * i, 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
            draw.circle(
                rc.screen, color,
                pos[i][[1, 0]] * tile_size + tile_size // 2 + rc.offset + (rc.size - self.minsize) / 2 + 0.5,
                tile_size // 3,
            )
            draw.rect(
                rc.screen, color,
                Rect(
                    goal[i][[1, 0]] * tile_size + tile_size / 6 + rc.offset + (rc.size - self.minsize) / 2 + 0.5,
                    (tile_size / 3 * 2, tile_size / 3 * 2),
                ),
                3,
            )
            for j in range(i):
                a = attention[i, j]
                if a > 0:
                    p1 = pos[i][[1, 0]] * tile_size + tile_size / 2 + rc.offset + (rc.size - self.minsize) / 2 + 0.5
                    p2 = pos[j][[1, 0]] * tile_size + tile_size / 2 + rc.offset + (rc.size - self.minsize) / 2 + 0.5
                    draw.line(rc.screen, [int(np.clip(a, 0, 1) * 255), 0, 0], p1, p2, int(a * tile_size / 2))
                    # distance = np.linalg.norm(p1 - p2)
                    # radius = distance / 2 / math.sin(CONFIG.COMM_ARC_ANGLE / 2)
                    # center = (p1 + p2) / 2 + np.array([p2[1] - p1[1], p1[0] - p2[0]]) / 2 / math.tan(CONFIG.COMM_ARC_ANGLE / 2)
                    # draw.aaline(rc.screen, "blue", (p1 + p2) / 2, center)
                    # angle = math.atan2(p2[1] - center[1], p2[0] - center[0])
                    # draw.arc(
                    #     rc.screen,
                    #     "blue",
                    #     Rect(
                    #         center - Vector2([radius, radius]),
                    #         Vector2([2 * radius, 2 * radius]),
                    #     ),
                    #     -angle - CONFIG.COMM_ARC_ANGLE,
                    #     -angle,
                    #     10,
                    # )
                    # draw.rect(
                    #     rc.screen,
                    #     "blue",
                    #     Rect(
                    #         center - Vector2([radius, radius]),
                    #         Vector2([2 * radius, 2 * radius]),
                    #     ),
                    #     2,
                    #     # angle,
                    #     # angle + CONFIG.COMM_ARC_ANGLE,
                    # )



    def min_size(self) -> Vector2:
        return Vector2(self.run.map_width, self.run.map_height) * CONFIG.TILE_SIZE

class AnchorWidget(Widget):
    def __init__(self, w: Widget, anchor: Vector2):
        self.w = w
        self.anchor = anchor

    def children(self):
        yield self.w

    def min_size(self):
        return self.w.minsize

    def draw(self, rc: RenderingContext) -> None:
        remaining = rc.size - self.w.minsize
        pos = remaining.elementwise() * self.anchor
        self.w.draw(rc.subrect_abs(Rect(pos, self.w.minsize)))


class TextWidget(Widget):
    def __init__(self, text: str, font_size: int = 30, font_color: Color = "white"):
        self.text = text
        self.font_size = font_size
        self.font_color = font_color
        self.font = pygame.font.SysFont(CONFIG.FONT_NAME or pygame.font.get_default_font(), self.font_size)

    def min_size(self) -> Vector2:
        return Vector2(self.font.size(self.text))

    def draw(self, rc: RenderingContext) -> None:
        text = self.font.render(self.text, True, self.font_color)
        rc.screen.blit(text, rc.offset + (rc.size - self.minsize) / 2)


class ButtonWidget(Widget):
    def __init__(self, text: str, action: typing.Callable[[], None] | None = None):
        self.text = text
        self.action = action
        self.clicked = False

    def min_size(self) -> Vector2:
        return Vector2(100, 50)

    def poll(self) -> bool:
        clicked = self.clicked
        self.clicked = False
        return clicked

    def draw(self, rc: RenderingContext) -> None:
        rect = Rect(rc.offset, rc.size)
        rc.add_hitbox(rect, self)
        draw.rect(rc.screen, "blue", rect)
        text = pygame.font.SysFont(CONFIG.FONT_NAME or pygame.font.get_default_font(), 30).render(self.text, True, "white")
        rc.screen.blit(text, rc.offset + (rc.size - Vector2(text.get_size())) / 2)

    def on_click(self) -> None:
        if self.action is not None:
            self.action()
        else:
            self.clicked = True

class GridWidget(Widget):
    def __init__(self, *widgets):
        self.widgets = widgets
        self.rows = len(widgets)
        self.cols = len(widgets[0])
        for row in widgets:
            assert len(row) == self.cols
            for widget in row:
                assert isinstance(widget, Widget) or widget is None

    def children(self):
        for row in self.widgets:
            for widget in row:
                if widget is not None:
                    yield widget

    def min_size(self) -> Vector2:
        row_sizes = [max(widget.minsize.y if widget else 0 for widget in row) for row in self.widgets]
        col_sizes = [max(widget.minsize.x if widget else 0 for widget in col) for col in zip(*self.widgets)]
        self.row_sizes = row_sizes
        self.col_sizes = col_sizes

        return Vector2(sum(col_sizes), sum(row_sizes))

    def draw(self, rc: RenderingContext) -> None:
        for i, row in enumerate(self.widgets):
            for j, widget in enumerate(row):
                if widget is not None:
                    widget.draw(rc.subrect_abs(Rect(
                        (sum(self.col_sizes[:j]), sum(self.row_sizes[:i])),
                        Vector2(self.col_sizes[j], self.row_sizes[i]),
                    )))

class SliderWidget(Widget):
    def __init__(
            self,
            min_: float, max_: float, value: float, step: float,
            action: typing.Callable[[float], None] | None = None,
            min_width: int = 100,
            height: int = 16,
        ):
        self.min = min_
        self.max = max_
        self.value = value
        self.step = step
        self.action = action
        self.min_width = min_width
        self.height = height

    def min_size(self) -> Vector2:
        return Vector2(self.min_width, self.height)

    def draw(self, rc: RenderingContext) -> None:
        rect = Rect(rc.offset, rc.size)
        self.rect = rect
        self.value = np.clip(self.value, self.min, self.max)
        rc.add_hitbox(rect, self)
        draw.rect(rc.screen, "black", rect)
        draw.rect(
            rc.screen,
            "white",
            Rect(
                rect.left + (self.value - self.min) / (self.max - self.min) * (rect.width - 2),
                rect.top,
                2,
                rect.height,
            ),
        )
        draw.rect(
            rc.screen,
            "white",
            Rect(
                rect.left,
                rect.top + rect.height // 2 - 1,
                rect.width,
                1,
            ),
        )

    def on_drag(self, delta):
        self.value = np.clip(self.value + delta.x / self.rect.width * (self.max - self.min), self.min, self.max)
        if self.action is not None:
            rounded = round(self.value / self.step) * self.step
            self.action(rounded)


def get_frame(run: Run, time):
    root = (run_widget := RunWidget(run))

    run_widget.time = time

    root.recompute_min_size()
    surf = pygame.Surface(root.minsize)
    surf.fill("black")
    rc = RenderingContext(surf, Vector2(root.minsize), 0.0)
    root.draw(rc)

    return pygame.image.tobytes(surf, "RGB"), root.minsize


def make_video(run, filepath: Path, duration: float, fps: int) -> None:
    frames = []
    frame_count = int(fps * duration)

    for i in tqdm.tqdm(range(frame_count)):
        time = i / (frame_count - 1) * int(len(run.positions) * 1.05 + 5)
        frame_bytes, size = get_frame(run, time)
        size = (int(size.x), int(size.y))

        img_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(size[1], size[0], 3)[:, :, ::-1]
        frames.append(img_array)

    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(filepath), fourcc, fps, (size[0], size[1]))
        for frame in frames:
            video.write(frame)

        video.release()


def application(run: Run):
    time = 0.0
    running = True
    pygame.font.init()
    pygame.init()
    clock = pygame.time.Clock()

    root = AnchorWidget(
            GridWidget(
                [run_widget := RunWidget(run)],
                [ButtonWidget("Pause/Unpause", lambda:  setattr(run_widget, "paused", not run_widget.paused))],
                [time_slider := SliderWidget(0, len(run.positions), 0.0, 0.5, lambda t: setattr(run_widget, "time", t))],
                [speed_slider := SliderWidget(-10, 10, 1.0, 0.1, lambda s: setattr(run_widget, "speed", s))],
            ),
        Vector2(0.5, 0.5),
    )

    root.recompute_min_size()

    size = root.minsize
    screen = pygame.display.set_mode(root.minsize, pygame.locals.RESIZABLE)
    dragged = None
    mouse_down = False
    last_frame_time = 0.0

    while running:
        click_pos = None
        mouse_delta = Vector2(pygame.mouse.get_rel())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.WINDOWRESIZED:
                size = (event.x, event.y)
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = Vector2(event.pos)
                if event.button == 1:
                    mouse_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False

        root.recompute_min_size()
        minsize = root.minsize
        size = (max(size[0], minsize.x), max(size[1], minsize.y))

        screen.fill("black")

        if not run_widget.paused:
            time_slider.value = run_widget.time

        rc = RenderingContext(screen, Vector2(size), time - last_frame_time)
        root.draw(rc)

        handled_click = False
        if click_pos is not None:
            for rect, widget in rc.hitboxes:
                if rect.collidepoint(click_pos):
                    widget.on_click()
                    dragged = widget
                    handled_click = True
                    break

        if not mouse_down:
            dragged = None

        if dragged is not None:
            dragged.on_drag(mouse_delta)

        last_frame_time = time
        time = pygame.time.get_ticks() / 1000
        pygame.display.flip()
        clock.tick(60)
        click_pos = None

    pygame.quit()


pygame.font.init()
pygame.init()

def main():
    parser = argparse.ArgumentParser(prog="visualize")
    subparsers = parser.add_subparsers(help="Action to perform", dest="action")

    parser_run = subparsers.add_parser("run", help="Run the visualizer in interactive mode")
    parser_run.add_argument("--map", type=Path, help="Path to the map file")
    parser_run.add_argument("--results", type=Path, help="Path to the results file")
    parser_run.add_argument("--run-id", type=int, help="ID of the run to visualize", default=0)

    parser_video = subparsers.add_parser("video", help="Generate a video of the run")
    parser_video.add_argument("--map", type=Path, help="Path to the map file")
    parser_video.add_argument("--results", type=Path, help="Path to the results file")
    parser_video.add_argument("--run-id", type=int, help="ID of the run to visualize", default=0)
    parser_video.add_argument("--output", type=Path, help="Path to the output video file", default="output.mp4")
    parser_video.add_argument("--duration", type=float, help="Duration of the resulting video", default=20.0)
    parser_video.add_argument("--fps", type=int, help="Frames per second", default=30)

    if len(sys.argv) == 1:

        RUN_ID = 0
        run = load_run(
            Path("DCC_test/test_set/32x32size_16agents_0.2density_30.pth"),
            Path("scrimp_test/results/32x32size_16agents_0.2density_30.pth"),
            RUN_ID)
        application(run)

        parser.print_help()
    else:
        args = parser.parse_args()
        print(args)
        if args.action == "run":
            run = load_run(args.map, args.results, args.run_id)
            application(run)
        elif args.action == "video":
            run = load_run(args.map, args.results, args.run_id)
            make_video(run, args.output, args.duration, args.fps)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
