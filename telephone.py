"""

A game of AI telephone. One agent gets a task and passes it to a chain of sub-agents. 
The last agent executes the task.

"""
import argparse 
import rich
import ollama

global MODEL, TASK_INSTRUCTION, MAX_DEPTH
MODEL = "qwen3.5:0.8b"
TASK_INSTRUCTION = "Say the phrase 'The quick brown fox jumps over the lazy dog'."
DELEGATION_INSTRUCTION = """\
You are an assistant that delegates tasks to a sub-agent. You have an invoke_subagent tool \
available that you should use to invoke the subagent.
It takes a single string argument `instructions` which should contain the instructions for the sub-agent. 

The sub-agent will perform the task based on the instructions you provide and will return a \
message when completed indicating completion. 

Do nothing else. Only use the invoke_subagent tool to delegate to the sub-agent."""

MAX_DEPTH = 10
MAX_TOOL_RETRIES = 3
CHAT_TIMEOUT = 45  # seconds per ollama call
DEBUG = False

client = ollama.Client(timeout=CHAT_TIMEOUT)

AGENT_COLORS = [
    "dodger_blue2", "deep_sky_blue4", "blue3", "navy_blue",
    "dodger_blue1", "deep_sky_blue1", "dark_turquoise", "cyan1",
    "blue_violet", "cornflower_blue", "steel_blue",
]
_agent_color_map: dict[str, str] = {}
_tool_call_count = 0
_agent_id_counter = 0


def _agent_color(agent_id: str) -> str:
    if agent_id not in _agent_color_map:
        _agent_color_map[agent_id] = AGENT_COLORS[len(_agent_color_map) % len(AGENT_COLORS)]
    return _agent_color_map[agent_id]


def _rp(agent_id: str, msg: str) -> None:
    color = _agent_color(agent_id)
    rich.print(f"[[{color}]{agent_id}[/{color}]]", end=" ")
    print(msg)

def _rp_inst(agent_id: str, msg: str) -> None:
    color = _agent_color(agent_id)
    rich.print(f"[[{color}]{agent_id}[/{color}]] {msg}")


def _run_final_call(agent_id: str, instructions: str) -> str:
    if DEBUG:
        _rp(agent_id, "Final tool call. Ending delegation loop after this.")
    response = client.chat(model=MODEL, messages=[{
        'role': 'user',
        'content': f"# Task\n{instructions}",
    }])
    print("-" * 60)
    _rp(agent_id, response.message.content)
    print("-" * 60)
    return ("COMPLETE", response.message.content)


def _run_delegating_call(agent_id: str, instructions: str) -> str:
    messages = [{'role': 'user', 'content': f"# Instruction\n{DELEGATION_INSTRUCTION}\n# Task\n{instructions}"}]
    response = client.chat(model=MODEL, messages=messages, tools=[invoke_subagent])

    last_result = None
    consecutive_errors = 0
    while response.message.tool_calls:
        messages.append(response.message)
        for tool_call in response.message.tool_calls:
            if DEBUG:
                _rp(agent_id, f"Received tool call: {tool_call.function.name} with arguments {tool_call.function.arguments}")
            if tool_call.function.name != 'invoke_subagent':
                continue
            try:
                if DEBUG:
                    _rp_inst(agent_id, f"Invoking sub-agent ...")
                result = invoke_subagent(**tool_call.function.arguments)
                last_result = result
                consecutive_errors = 0
            except TypeError as e:
                consecutive_errors += 1
                if consecutive_errors >= MAX_TOOL_RETRIES:
                    _rp_inst(agent_id, f"[bold red]Aborting[/bold red]: {consecutive_errors} consecutive tool call errors. Last error: {e}")
                    return ("FAILED", None)
                result = f"Error: {e}. Please call invoke_subagent again with the correct arguments."
            messages.append({'role': 'tool', 'content': str(result)})
        if _tool_call_count >= MAX_DEPTH:
            break
        response = client.chat(model=MODEL, messages=messages, tools=[invoke_subagent])

    return ("COMPLETE", last_result)


def invoke_subagent(instructions: str) -> str:
    """Delegate a task to a sub-agent.

    Args:
    instructions: A string containing the instructions for the sub-agent.

    Returns:
    A string indicating the result of the sub-agent invocation.
    """
    global _agent_id_counter, _tool_call_count

    _agent_id_counter += 1
    agent_id = f"subagent-{_agent_id_counter}"
    _rp_inst(agent_id, f"Invoked with prompt: [dim]{instructions}[/dim]")

    _tool_call_count += 1
    if DEBUG:
        _rp(agent_id, f"Tool call count: {_tool_call_count}")

    if _tool_call_count > MAX_DEPTH:
        _rp(agent_id, f"Maximum tool call limit of {MAX_DEPTH} reached. Ending delegation loop.")
        return "Maximum tool call limit reached."

    if _tool_call_count == MAX_DEPTH:
        status, response = _run_final_call(agent_id, instructions)
        if status != "COMPLETE":
            _rp(agent_id, f"Final call did not complete successfully. Status: {status}. Response: {response}")
        return response

    status, response = _run_delegating_call(agent_id, instructions)
    return response


def play_telephone(model, depth, task) -> str:
    global MODEL, MAX_DEPTH, TASK_INSTRUCTION, _tool_call_count, _agent_id_counter
    MODEL = model
    MAX_DEPTH = depth
    TASK_INSTRUCTION = task
    _tool_call_count = 0
    _agent_id_counter = 0
    response = invoke_subagent(TASK_INSTRUCTION)
    return response


def main():
    global MODEL, TASK_INSTRUCTION, MAX_DEPTH
    # CLI parser
    parser = argparse.ArgumentParser(description="Run the telephone experiment with configurable parameters.")
    parser.add_argument("--model", type=str, default=MODEL, help="The model to use for the experiment (default: qwen3.5:0.8b).")
    parser.add_argument("--task", type=str, default=TASK_INSTRUCTION, help="The instruction to give to the initial agent.")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH, help="The maximum depth of delegation (i.e., how many times the agent can call the tool to invoke a sub-agent).")

    args = parser.parse_args()
    MODEL = args.model
    TASK_INSTRUCTION = args.task
    MAX_DEPTH = args.depth
    invoke_subagent(TASK_INSTRUCTION)

if __name__ == "__main__":
    main()
