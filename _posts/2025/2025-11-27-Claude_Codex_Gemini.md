---
title: AI for Coding - Claude Codex Gemini CLI
date: 2025-11-27 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - AI
  - Claude
  - Coding
  - Copilot
  - LLM
  - Tools
  - VScode
---




<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## Introduction

I have experienced the AI coding for three stages.  
Stage 1:  vs code plugin, started from openai codex, then other LLM plugin like ClaudeDev, Cline 
Stage 2:  customer GUI like Cursor, Windsurf
Stage 3:  command line based, CLI, Anthropic Claude Code, Openai codex, Gemini code.


## CLI Installation

**OpenAI Codex CLI Installation:** The official method is via npm. On Linux/macOS, you can install globally with:

`npm install -g @openai/codex`

This places the `codex` command in your PATH[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=If%20you%E2%80%99re%20using%20macOS%20or,Linux%2C%20install%20it%20globally%20with). If you don’t have root, using **nvm** to manage Node lets you do `-g` installs in user space. On macOS or Linux, you can also use **Homebrew** as an alternative: for example, `brew install codex` is supported for a one-command installation[zitniklab.hms.harvard.edu](https://zitniklab.hms.harvard.edu/ToolUniverse/guide/building_ai_scientists/codex_cli.html#:~:text=GPT%20Codex%20CLI%20,Verify). If you cannot or prefer not to install globally, you can run Codex CLI with **npx** (which fetches and executes the package without persistent install). For example:

`npx @openai/codex [command]`

This will download the CLI package on-the-fly and run it. After installation, set your OpenAI API key in the environment:

**Anthropic Claude Code CLI Installation:** Claude’s CLI is also distributed via npm. Install it globally with:

`npm install -g @anthropic-ai/claude-code`

(The package name might also be `@anthropic/claude-cli` in earlier versions[educative.io](https://www.educative.io/blog/claude-code-vs-codex-vs-gemini-code-assist#:~:text=It%20runs%20as%20a%20CLI,all%20from%20natural%20language%20instructions), but the latest is as above.) Like with Codex, using nvm or a user-local Node setup will let you do this without root. Once installed, run `claude` in any project folder to start a session[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=Run%20the%20following%20command%20in,your%20terminal)[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=). The first time, it will prompt you to log in or provide an API key. You should have your **Anthropic API Key** ready (after signing up for Claude API or Claude Pro). Set the key as an environment variable for convenience:

`export CLAUDE_API_KEY="your-api-key-here"`

(Or you might be asked to paste it at prompt.) After launching, you’ll see a welcome screen possibly asking to choose a theme, then you’ll be in a chat-like interface in the terminal. Claude CLI doesn’t require root for any of this. On macOS, Homebrew might not have an official formula, but you could use `brew install anthropic-cli` if a community cask exists. Otherwise, npm or npx is the way. Using **npx**:

`npx @anthropic-ai/claude-code`

**Google Gemini CLI Installation:** You have multiple options and it’s quite flexible:

- **Quick run without install:** Use **npx** to run directly from GitHub:
    
    `npx https://github.com/google-gemini/gemini-cli`
    
    This fetches and executes the CLI from the repository (ensuring you get the latest version)[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=Step%202%3A%20Quickstart). This method is great if you don’t want to install anything system-wide – no root needed, just requires Node and internet access.
    
- **Global install via npm:**
    
    `npm install -g @google/gemini-cli`
    
    Then launch with `gemini`[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=%60npx%20https%3A%2F%2Fgithub.com%2Fgoogle). Again, with nvm or user Node, no root is needed. After running `gemini` the first time, it will prompt you to log in. You can choose to authenticate with your Google account (a browser-based OAuth flow) or use a Gemini API key[codeant.ai](https://www.codeant.ai/blogs/claude-code-cli-vs-codex-cli-vs-gemini-cli-best-ai-cli-tool-for-developers-in-2025#:~:text=Launching%20,followed%20by%20a%20login%20method). For most users, logging in with Google is simplest – it ties into the free tier usage. If you have a Google Cloud project and obtained an API key (from Google AI Studio) for Gemini, you can export it:



## Gemini installation Debug

Intel PC with GTX 1080 WSL2.

用 web authorize, check 
~/.gemini/settings.json
{
  "security": {
    "auth": {
      "selectedType": "oauth-personal"
    }
  }
}

AMD PC with RTX 3060 WSL2.

用 web authorize, check 
~/.gemini/settings.json
{
  "security": {
    "auth": {
      "selectedType": "gemini-api-key"
    }
  }
}

另外在 .basrhc 要設定 GEMINI_API_KEY
可以在這個 website get API KEY
https://aistudio.google.com/app/api-keys

