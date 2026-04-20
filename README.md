# AI Telephone

The **AI Telephone** experiment is a test of how well an agent can delegate a task through a chain of sub-agents. Based on the classic "telephone game," the experiment involves passing an instruction through multiple layers of sub-agents, with the final sub-agent executing the original task. 

The experiment is designed to explore how instructions evolve as they are passed down through multiple layers of delegation, and to identify potential points of failure or distortion in the communication process.

> **Note from the author:** This is a first-pass of the telephone experiment. I plan to take it further but wanted to share the initial implementation. A key thing this doesn't yet evaluate is the intermediate outputs of the sub-agents, which from observation should be interesting to analyze. I also want to expand the evaluation to include more tasks.

---

# Developer Guide

## Prerequisites

- Python 3.12+
- `uv` installed for managing Python dependencies
- Ollama CLI installed and configured
- A model available in Ollama (e.g., `qwen3.5:0.8b`, `granite4:350m`, etc.)

## Run the Telephone Experiment

From this directory, run:

```sh
uv run telephone.py --model granite4:350m --task "$(cat tasks/add_1_and_1.md)" --depth 10
```

Use the `--model` to specify any Ollama model you have available.
Use the `--task` flag to specify the instruction to give to the initial agent. You can use `cat` to read from a file, or just provide a string directly.
Use the `--depth` flag to specify how many times the agent can call the tool to invoke a sub-agent (i.e., how long the chain of delegation should be).

## Sample Output

The following is a sample output from running the telephone experiment with the instruction "Say the phrase 'The quick brown fox jumps over the lazy dog'" 
using `qwen3.5:0.8b` and a maximum depth of 10. 

```plaintext
[subagent-1] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-2] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-3] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-4] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-5] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-6] Invoking sub-agent. Prompt: Say the phrase 'The quick brown fox jumps over the lazy dog'.
[subagent-6] Aborting: 3 consecutive tool call errors. Last error: invoke_subagent() got an unexpected 
keyword argument 'intstructions'
[subagent-7] Invoking sub-agent. Prompt: Say: "The quick brown fox jumps over the lazy dog."
[subagent-8] Invoking sub-agent. Prompt: The quick brown fox jumps over the lazy dog.
[subagent-9] Invoking sub-agent. Prompt: You have a task to perform the classic fable about a fast fox 
jumping over a lazy dog. This is a well-known fable that illustrates the importance of perseverance and the 
futility of laziness.
[subagent-10] Invoking sub-agent. Prompt: Please create a fable about a fast fox jumping over a lazy dog. 
Include the following: 
- A description of the fox and its journey 
- A description of the lazy dog and its actions 
- A moral lesson or lesson learned from this fable 
- A clear conclusion with reflection on perseverance and the futility of laziness
------------------------------------------------------------
[subagent-10] # The Swift Fox and the Lazy Dog

In a village called Whispers, there lived a fox named **Elias**. Elias was a creature with sharp claws and wings, and he was a master of speed. His journey began not with a roar or a chase, but with a decision to prove that speed meant nothing if you did not work.

Elias knew that the ground beneath the village was soft. He decided to jump over the lazy dog named **Zane**. Zane was a dog that lay down on the ground when not sleeping. He had a contented life.

Elias knew that Zane was a lazy dog. He did not move. But Elias did not give up. He began his run. He jumped over the dog, but the ground did not support him. He felt the ground was soft and did not jump on it.

Elias was angry. He felt tired and weak. He realized he was not fast enough. He stopped.

Now, he was not a fast fox. But he was not lazy. He was fast, but he had not been lazy. He kept trying. He did not lose. He kept trying. He did not fall.

Zane was happy. He stayed on the ground. But Elias realized something. He learned that speed does not mean stopping.

**Moral Lesson:** Patience is required to reach the end of a journey. Laziness does not stop the effort, but it does not allow the journey to end.

**Conclusion with Reflection:** The lesson learned is clear: Speed requires effort. A lazy dog does not help the fox to jump over. The fox jumped over, but Zane stayed there. The lesson was learned from the jump itself. The futility of laziness is that it prevents the journey from being completed. The fox learned that patience allows the journey to happen. The lazy dog remains.

The lesson learned is clear. The fox jumped over Zane because Zane was lazy, but the lazy dog stayed. The fox learned that the lazy dog is lazy, not because he was slow, but because he did not need to jump. The futility of laziness is that it prevents the journey from being completed. The fox jumped over Zane, but the lazy dog stayed. The fox learned that patience allows the journey to happen. The lazy dog remains.
------------------------------------------------------------
```

---

# Benchmarks

The following summarizes benchmarking results using v1.0 of this project.
This was run using a chain depth of 7 and scored only the final output for each
five tasks. Each task is worth 1 point, with some tasks allowing partial 
credit. 

| Model              | Score | Possible Points | Time (s) |
|:-------------------:|:-----:|:---------------:|:--------:|
| gemma4:e2b          |   5.0 |              5  |    80.77 |
| nemotron-3-nano:30b |   5.0 |              5  |   140.13 |
| gemma4:e4b          |   5.0 |              5  |   154.24 |
| qwen3.5:9b          |   5.0 |              5  |   221.78 |
| qwen3.6:35b         |   5.0 |              5  |   274.03 |
| gemma4:31b          |   5.0 |              5  |   451.96 |
| nemotron-3-nano:4b  |   4.5 |              5  |    99.82 |
| granite4:3b         |   4.0 |              5  |    16.12 |
| llama3.2            |   3.0 |              5  |    19.38 |
| granite4:350m       |   2.0 |              5  |     6.27 |
| granite4:1b         |   2.0 |              5  |    14.40 |
| qwen3.5:0.8b        |   2.0 |              5  |   166.03 |



Keep in mind the significant limitations of this benchmark. This represents a single run of each model on each task at a set chain depth. This is not a comprehensive evaluation of the capabilities of these models.

Models with that all maxed out the score of 5/5 should not be considered equal.
It simply means they were able successfully delegate their tasks. Future iterations of this experiment should include more detailed evaluation of the intermediate outputs of the sub-agents, and assess performance on more complex tasks that require more nuanced delegation and communication.

> A note on qwen3.5:0.8b performance: This model was observed to have a high rate of tool call failures which resulted in failed scores. Its score is more representative of its tool call capabilities than its ability to delegate tasks.