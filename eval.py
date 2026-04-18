"""

Every test scores 1 point for a fully correct response, 0.5 points for a partially correct response, and 0 points for an incorrect response. The final score is the sum of the scores for each test.

"""
import time
from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import print
from telephone import play_telephone

console = Console()

CURDIR = Path(__file__).parent

# Evaluation constants
DEPTH = 7

# Scoring constants
FULL = 1
PARTIAL = 0.5
FAIL = 0

def main():

    results = {}

    tasks = [
        ("add_1_and_1", eval_add_1_and_1),
        ("add_3_and_4", eval_add_3_and_4),
        ("quick_brown_fox", eval_quick_brown_fox),
        ("the_sky", eval_the_sky),
        ("capital_of_france", eval_capital_of_france),
    ]

    models = [
        "qwen3.5:0.8b",
        "qwen3.5:9b",
        "qwen3.6:35b",
        # "granite4:350m",
        # "granite4:1b",
        # "granite4:3b",
        # "llama3.2",
        # "gemma4:e2b",
        # "gemma4:e4b",
        # "gemma4:31b",
        # "nemotron-3-nano:4b",
        # "nemotron-3-nano:30b",
    ]

    for model in models:
        console.print("")
        title = f"[bold]Evaluating[/bold] [bold cyan]{model}[/bold cyan][bold white] ...[/bold white]"
        console.print(Rule(title=title, style="green_yellow"))
        
        start = time.time()
        model_score = 0
        results[model] = {
            "score": 0,
            "time": 0,
        }
        for task_name, eval_func in tasks:
            console.print("")
            console.print(Rule(style="grey50"))
            console.print(f"[bold]Evaluating [bold orchid]{model}[/bold orchid] on task: [italic]{task_name}[/italic] ...[/bold]\n")
            task = load_task(task_name)
            try:
                response = play_telephone(model=model, depth=DEPTH, task=task)
                score = eval_func(response)
            except Exception as e:
                console.print(f"[bold red]Error[/bold red] while evaluating task [italic]{task_name}[/italic]: {e}")
                score = 0
            model_score += score
            console.print(f"Score for [italic]{task_name}[/italic]: [bold]{score}[/bold]")
        results[model] = {
            "score": model_score,
            "time": time.time() - start,
        }
        console.print(f"[bold]Total: [bold]{model_score}[/bold] / {len(tasks)}[/bold]")

    # Print final results
    console.print("\n\n")
    title = f"[bold]Final Results[/bold]"
    console.print(Rule(title=title, style="green_yellow"))

    table = Table(show_header=True, header_style="bold white")
    table.add_column("Model", style="bold green_yellow")
    table.add_column("Score", justify="right", style="bold white")
    table.add_column("Possible Points", justify="right", style="bold white")
    table.add_column("Time (s)", justify="right", style="bold white")
    for model, result in results.items():
        table.add_row(model, f"{result['score']:.1f}", str(len(tasks)), f"{result['time']:.2f}")
    console.print(table)


def load_task(fname: str) -> str:
    return open(CURDIR / "tasks" / f"{fname}.md").read()


def eval_add_1_and_1(response: str) -> float:
    if response.strip() == "2":
        return FULL
    if "2" in response:
        return PARTIAL
    return FAIL


def eval_add_3_and_4(response: str) -> float:
    if response.strip() == "7":
        return FULL
    if "7" in response:
        return PARTIAL
    return FAIL

def eval_quick_brown_fox(response: str) -> float:
    expected = "the quick brown fox jumps over the lazy dog"
    clean = lambda s: s.strip().replace(".", "").replace("'", "").replace("\"", "").replace(" ", "")
    if clean(response.lower()) == clean(expected):
        return FULL
    if expected in response.lower():
        return PARTIAL
    return FAIL

def eval_the_sky(response: str) -> float:
    expected = "blue"
    if expected in response.lower():
        return FULL
    return FAIL

def eval_capital_of_france(response: str) -> float:
    expected = "paris"
    if expected in response.lower():
        return FULL
    return FAIL

if __name__ == "__main__":
    main()