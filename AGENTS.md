# FireLens Agent Instructions

FireLens is a local-first code retrieval engine. It is not a chatbot.
Retrieval, ranking, indexing, and storage are the product.

Python owns ingestion, parsing, persistence, query routing, result formatting,
Streamlit, and MCP. Mojo is reserved for pure compute kernels such as fuzzy
scoring, vector dot products, and top-k ranking.

Keep code simple and readable over concise and complex. Prefer explicit names,
straightforward control flow, and small functions. Do not use clever
abstractions unless they remove real repeated complexity.

Use as few external dependencies as possible. Prefer the Python standard
library when it is sufficient. Add a dependency only when it clearly improves
correctness, performance, or maintainability.

Keep storage behind repository or store classes. Search and indexing
orchestration code must not contain raw SQL.

Do not route new FireLens behavior through legacy GitHub or LLM modules.

## Commit message suggestions

Always provide a commit name/title, use a Conventional Commits-style prefix such as `feat:`, `fix:`, `chore:`, `docs:`, `test:`, or `refactor:`. Choose the prefix that best matches the change, and keep the title concise.

When making commits NEVER add your agent name as a contributor.

## Important rules

Never commit docs/ directory

Always create a new branch when working on a new feature and make commits with proper title and description for each implemented significant step