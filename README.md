# FireLens 🔥

Have you ever struggled to understand the source code of a complex library, framework, or tool you wanted to use? Navigating through unfamiliar codebases can be a challenge, especially when dealing with large projects or code written by others. That's where FireLens comes in – your AI-powered code comprehension assistant.

Whether you're a developer trying to understand a popular open-source project, a researcher exploring a new codebase, or a student learning from real-world code examples, FireLens empowers you to quickly grasp the inner workings of any codebase, saving you countless hours of manual code review and exploration.

> **TL;DR**
>
> FireLens is an AI-powered code analysis tool that makes it easier to understand the contents of any public GitHub repository.

## Features

- **Code Comprehension**: Ask questions about a GitHub repository's codebase, and FireLens will provide clear explanations, powered by OpenAI's GPT-3.5-turbo language model.
- **GitHub Integration**: Seamlessly fetch and analyze code directly from any public GitHub repository.
- **User-friendly Interface**: Interact with FireLens through a simple and intuitive Streamlit web interface.
- **Supported Languages**: Python (more to come).

## Getting Started

### Prerequisites

- Python 3.7 or later
- An OpenAI API key (for LLM functionality)
- GitHub Authentication Token (Optional, increases API rate limit)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aslanbayli/firelens.git
   cd firelens
   ```

2. Install all of the required packages:

    ```bash
    pip install -r requirements.txt
    ```
    
3. Add you dev keys to the `.env` file

4. Start the server:

    ```bash
    make server
    ```

5. Start the client (in a separate terminal):

    ```bash
    make client
    ```

After the last command, a new tab in your default browser will open.
![image](https://github.com/aslanbayli/firelens/assets/48028559/78d438ee-723c-4e14-82b1-fcb8174bc433)

## Potential improvements
- Add support for repositories written in programming langauges other than Python
- Ability to choose between different LLMs (Currently supports GPT-3.5-turbo)
- More complex actions such as getting information about open issues on the repo, or looking up additional information using StackOverflow.

### Example usage
![image](https://github.com/aslanbayli/firelens/assets/48028559/1a19f38e-9483-4ec9-ad49-5bf0e3c49198)


