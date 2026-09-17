# Data Structures & Algorithms (DSA)

A categorized collection of Data Structures & Algorithms problems and solutions in **Python**, organized by topic for easy navigation and practice.

## 📂 Categories

| Category | Description |
|---|---|
| [Array](./Array) | Traversal, two pointers, sliding window, prefix sums |
| [Linked List](./Linked_List) | Singly, doubly, and circular linked lists |
| [Stack](./Stack) | Balanced parentheses, monotonic stacks, expression evaluation |
| [Queue](./Queue) | Queue, deque, circular queue |
| [Trees](./Trees) | Binary trees, BSTs, tries, traversals |
| [Graphs](./Graphs) | BFS, DFS, shortest paths, MST, topological sort |
| [Hashing](./Hashing) | Hash maps/sets, frequency counting, grouping |
| [Sorting](./Sorting) | Sorting algorithms and applications |
| [Searching](./Searching) | Binary search and variants |
| [Dynamic Programming](./Dynamic_Programming) | 1D/2D DP, knapsack, LIS, LCS |
| [Recursion](./Recursion) | Recursion and divide-and-conquer |
| [Strings](./Strings) | String manipulation and pattern matching |
| [Heap / Priority Queue](./Heap_Priority_Queue) | Kth largest/smallest, merge k lists, scheduling |
| [Greedy](./Greedy) | Greedy algorithms and intuition |
| [Backtracking](./Backtracking) | Permutations, combinations, N-Queens, Sudoku |
| [Bit Manipulation](./Bit_Manipulation) | Bitwise tricks and problems |
| [Math](./Math) | Number theory and math-based problems |

## 🗂 Structure

Each category folder contains:
- `README.md` — a problem list/tracker table and notes section for that topic
- `template.py` — a starter template for adding new solutions

Example layout:
```
DSA/
├── Array/
│   ├── README.md
│   └── template.py
├── Linked_List/
│   ├── README.md
│   └── template.py
├── ...
└── README.md
```

## ✍️ How to Add a Problem

1. Copy `template.py` in the relevant category folder and rename it (e.g. `two_sum.py`).
2. Fill in the problem statement, approach, and complexity in the docstring.
3. Implement your solution and add test cases under `if __name__ == "__main__":`.
4. Add a row for it in that category's `README.md` problem table, with a link to the file.

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/DSA.git
cd DSA
python3 Array/template.py
```

## 📌 Roadmap

- [ ] Add solved problems to each category
- [ ] Add time/space complexity notes
- [ ] Add difficulty tags (Easy / Medium / Hard)
- [ ] Add unit tests

## 🤝 Contributing

Contributions are welcome! Feel free to open a PR with new problems, cleaner solutions, or better notes.

## 📄 License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
