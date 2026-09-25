# Beam Search Algorithm

Beam Search is a heuristic search algorithm that explores a search space by keeping only the most promising candidates at each step. It is commonly used in Artificial Intelligence, Natural Language Processing (NLP), Speech Recognition, Machine Translation, and Large Language Models (LLMs). Instead of exploring all possible paths like Breadth-First Search (BFS), Beam Search retains only the top **k** candidates, known as the **beam width**. :contentReference[oaicite:0]{index=0}

--------------------------------------------------

## Key Idea

At every level:

1. Generate all possible successor nodes.
2. Score each node using a heuristic or probability.
3. Keep only the top **k** nodes.
4. Discard the remaining nodes.
5. Repeat until a goal state is reached.

```text
Beam Width (k) = Number of candidates retained at each level
```

A larger beam width explores more possibilities but requires more computation and memory. A smaller beam width is faster but may miss better solutions. :contentReference[oaicite:1]{index=1}

--------------------------------------------------

## How Beam Search Works

Suppose:

```text
Beam Width (k) = 2
```

Initial State:

```text
Start
```

Generate successors:

```text
A (Score = 8)
B (Score = 6)
C (Score = 4)
```

Keep only the top 2:

```text
A
B
```

Expand:

```text
A → D (9), E (7)
B → F (8), G (5)
```

Candidates:

```text
D (9)
F (8)
E (7)
G (5)
```

Keep top 2:

```text
D
F
```

Continue until the goal is reached. :contentReference[oaicite:2]{index=2}

--------------------------------------------------

## Beam Search Pseudocode

```text
BeamSearch(start, beam_width):

    beam = [start]

    while beam is not empty:

        candidates = []

        for node in beam:
            expand node
            add successors to candidates

        sort candidates by score

        beam = top beam_width candidates

        if goal found:
            return solution

    return failure
```

--------------------------------------------------

## Beam Search for Sequence Generation

Beam Search is widely used in NLP and LLMs.

At each decoding step:

```text
Current Sequence
      ↓
Generate Next Tokens
      ↓
Score All Sequences
      ↓
Keep Top-k Sequences
      ↓
Repeat
```

Example:

```text
Beam Width = 3

Step 1:
"The"      0.50
"A"        0.30
"This"     0.20

Step 2:
"The cat"      0.45
"The dog"      0.40
"A cat"        0.35

Keep Top 3
```

The algorithm continues until an End-of-Sequence (EOS) token is generated. :contentReference[oaicite:3]{index=3}

--------------------------------------------------

## Beam Width (k)

| Beam Width | Behavior |
|------------|----------|
| 1 | Greedy Search |
| 2-5 | Fast and Efficient |
| 10-20 | Better Accuracy |
| Large k | Similar to Best-First Search |
| Infinite | Equivalent to BFS/Exhaustive Search |

A beam width of 1 behaves like greedy decoding, while very large beam widths approach exhaustive search. :contentReference[oaicite:4]{index=4}

--------------------------------------------------

## Time Complexity

Let:

```text
b = Branching Factor
k = Beam Width
d = Search Depth
```

Time Complexity:

```text
O(k × b × d)
```

Space Complexity:

```text
O(k)
```

Beam Search is significantly more memory-efficient than Breadth-First Search because only k candidates are retained at each level. :contentReference[oaicite:5]{index=5}

--------------------------------------------------

## Advantages

- Faster than exhaustive search
- Memory efficient
- Easy to implement
- Scales to large search spaces
- Widely used in NLP and AI systems
- Provides a balance between speed and accuracy

--------------------------------------------------

## Limitations

- Not guaranteed to find the optimal solution
- May discard promising paths too early
- Performance depends heavily on the heuristic function
- Sensitive to beam width selection
- Larger beam widths increase computational cost

Because paths are pruned during the search, Beam Search is generally neither complete nor optimal. :contentReference[oaicite:6]{index=6}

--------------------------------------------------

## Beam Search vs Other Search Algorithms

| Feature | BFS | DFS | Greedy | Beam Search |
|----------|-----|-----|---------|-------------|
| Memory Usage | High | Low | Low | Moderate |
| Optimal Solution | Yes | No | No | No |
| Complete | Yes | No | No | No |
| Heuristic Based | No | No | Yes | Yes |
| Scalable | Poor | Good | Good | Excellent |

--------------------------------------------------

## Applications

### Natural Language Processing

- Machine Translation
- Text Summarization
- Text Generation
- Language Modeling

### Speech Recognition

- Voice Assistants
- Speech-to-Text Systems

### Robotics

- Path Planning
- Navigation Systems

### Game AI

- Move Selection
- Strategy Optimization

### Large Language Models

- Chatbots
- Question Answering
- Content Generation

Beam Search is a common decoding strategy in sequence-generation systems and language models. :contentReference[oaicite:7]{index=7}

--------------------------------------------------

## Best Practices

1. Start with beam width between 3 and 10.
2. Use high-quality heuristic functions.
3. Monitor memory usage for large beam widths.
4. Apply length normalization in NLP tasks.
5. Compare results against greedy decoding.

--------------------------------------------------

## Conclusion

Beam Search is a heuristic search algorithm that balances search quality and computational efficiency by exploring only the most promising candidates at each level. It is widely used in machine translation, speech recognition, path planning, and modern LLM decoding because it provides better results than greedy search while remaining computationally tractable for large search spaces.

[Beam Search Implementation](./Beam%20Search.ipynb)