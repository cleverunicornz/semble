from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from semble.version import __version__

_HOME = Path.home()


def _exists_or_denied(path: Path) -> bool:
    """Distinguish between existence and permission issues."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    # PermissionError is a subclass of OSError
    # which is why this looks the way it does.
    except PermissionError:
        return True
    except OSError:
        return False
    return True


Action = Literal["created", "updated", "unchanged", "not-found", "removed", "error", "skipped"]
Mode = Literal["install", "uninstall"]


class IntegrationType(str, Enum):
    """Identifier for one of semble's install/uninstall integrations."""

    MCP = "mcp"
    INSTRUCTIONS = "instructions"
    SUBAGENT = "subagent"


SEMBLE_START = "<!-- SEMBLE_START -->"
SEMBLE_END = "<!-- SEMBLE_END -->"


def semble_pin() -> str:
    """Return the uvx --from specifier for the semble MCP server."""
    try:
        raw = importlib.metadata.distribution("semble").read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            url = data.get("url", "")
            if "dir_info" in data and url.startswith("file://"):
                path = url2pathname(urlparse(url).path)
                return f"{path}[mcp]"
            vcs_info = data.get("vcs_info", {})
            if vcs_info.get("vcs") == "git" and vcs_info.get("commit_id"):
                return f"git+{url}@{vcs_info['commit_id']}#egg=semble[mcp]"
    except Exception:
        pass
    return f"semble[mcp]=={__version__}"


SEMBLE_PIN = semble_pin()

_STDIO_SERVER_CONFIG: dict[str, object] = {
    "command": "uvx",
    "args": ["--from", SEMBLE_PIN, "semble"],
    "type": "stdio",
}

_OPENCODE_SERVER_CONFIG: dict[str, object] = {
    "command": ["uvx", "--from", SEMBLE_PIN, "semble"],
    "type": "local",  # opencode uses "local"/"remote", not "stdio"
    "enabled": True,
}

_BARE_STDIO_SERVER_CONFIG: dict[str, object] = {  # Windsurf: command/args only, no "type"
    "command": "uvx",
    "args": ["--from", SEMBLE_PIN, "semble"],
}

_ZED_SERVER_CONFIG: dict[str, object] = {  # Zed: command/args only, no "source"
    "command": "uvx",
    "args": ["--from", SEMBLE_PIN, "semble"],
}

INSTRUCTIONS = f"""\
{SEMBLE_START}
## Semble Semantic Search

A context-bound `semble` MCP server is available with two tools:
- `mcp__semble__semantic_search` — search the current agent workspace.
- `mcp__semble__semantic_index_status` — inspect exact index coverage, freshness, and changed paths.

The server already knows this agent's worktree, repository identity, and immutable starting revision. Never send or
guess a repository path.

`semantic_search` defaults to `facet="workspace"`, the normal choice. It searches two independent indexes and returns
two explicit sections: `changed_results` from modified/added/renamed files in the private delta, and
`unchanged_results` from untouched files in the immutable baseline. Modified, renamed, and deleted paths shadow their
old baseline versions in this effective current-workspace view.

Use a narrower facet only when it removes useful noise:
- `changed` — current changed-file delta only.
- `unchanged` — untouched baseline paths only.
- `base` — complete original snapshot, including old versions of files later modified, renamed, or deleted.

Every search synchronizes Git-visible writes before reading, so the first call after an edit is read-your-writes.
Large edit or formatter batches can make that call wait while the delta publishes; `index_context` reports the
generation, synchronization time, batching policy, indexed content classes, counts, and exclusions. Paths outside the
current worktree, ignored files, unsupported/binary/empty/oversized files, and unselected content classes are not
searched.

### Workflow

1. Call `mcp__semble__semantic_search` with a focused behavior, symbol, or code query. Omit `facet` for the combined
   current workspace.
2. Read `changed_results` and `unchanged_results` as separate ranked facets; use each result's `origin` and `change`
   fields as provenance.
3. Navigate directly to the returned file and line. Do not grep for the same content again.
4. After editing or formatting, call semantic search normally; it synchronizes pending delta changes itself.
5. Use `facet="changed"` to search only task-local work, `facet="unchanged"` for supporting untouched code, or
   `facet="base"` to compare with the original snapshot.
6. Set `content="docs"`, `content="config"`, or `content="all"` only when searching beyond code.
7. Call `mcp__semble__semantic_index_status` when you need full changed-path details or to verify what is indexed.
8. Use Grep only when you need every literal occurrence rather than semantic discovery.

For CLI fallback or subprocesses without MCP access, use `semble search`; CLI search is explicit repository search and
does not provide the context-bound baseline/delta facets.
If `semble` is not on `PATH`, use `uvx --from "{SEMBLE_PIN}" semble search ...`.
{SEMBLE_END}
"""


@dataclass(frozen=True)
class McpConfig:
    """MCP integration config for one agent."""

    path: Path
    key: str
    entry: dict[str, object]
    format: Literal["json", "toml"] = "json"


@dataclass(frozen=True)
class WriteResult:
    """Result of a single file write operation."""

    path: Path
    action: Action


@dataclass(frozen=True)
class AgentTarget:
    """Configuration for a single coding agent integration target."""

    id: str
    display_name: str
    binary: str | None  # for shutil.which detection
    config_dir: Path | None  # directory existence check for detection
    mcp: McpConfig | None
    instructions_path: Path | None  # None = not supported for this agent
    subagent_path: Path | None = None  # global (user-level) sub-agent file; None = unsupported

    def resolved_mcp_path(self) -> Path | None:
        """Return the resolved MCP config path, or None if MCP is unsupported."""
        return self.mcp.path if self.mcp else None


def _opencode_mcp_path() -> Path:
    """Return the opencode config path, preferring .jsonc over .json."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "opencode" if xdg else _HOME / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    json_ = base / "opencode.json"
    return jsonc if _exists_or_denied(jsonc) else (json_ if _exists_or_denied(json_) else jsonc)


def _vscode_mcp_path() -> Path:
    """Return the user-level VS Code mcp.json path for the current OS."""
    if sys.platform == "darwin":
        base = _HOME / "Library" / "Application Support" / "Code" / "User"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", _HOME)) / "Code" / "User"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", _HOME / ".config")) / "Code" / "User"
    return base / "mcp.json"


AGENTS: list[AgentTarget] = [
    AgentTarget(
        id="claude",
        display_name="Claude Code",
        binary="claude",
        config_dir=_HOME / ".claude",
        mcp=McpConfig(_HOME / ".claude.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".claude" / "CLAUDE.md",
        subagent_path=_HOME / ".claude" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="cursor",
        display_name="Cursor",
        binary="cursor",
        config_dir=_HOME / ".cursor",
        mcp=McpConfig(_HOME / ".cursor" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=None,  # Cursor instructions are project-local .mdc files
        subagent_path=_HOME / ".cursor" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="gemini",
        display_name="Gemini CLI",
        binary="gemini",
        config_dir=_HOME / ".gemini",
        mcp=McpConfig(_HOME / ".gemini" / "settings.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="kiro",
        display_name="Kiro",
        binary="kiro",
        config_dir=_HOME / ".kiro",
        mcp=McpConfig(_HOME / ".kiro" / "settings" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".kiro" / "steering" / "semble.md",
        subagent_path=_HOME / ".kiro" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="opencode",
        display_name="Opencode",
        binary="opencode",
        config_dir=_HOME / ".config" / "opencode",
        mcp=McpConfig(_opencode_mcp_path(), "mcp", _OPENCODE_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "opencode" / "AGENTS.md",
        subagent_path=_HOME / ".config" / "opencode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="copilot",
        display_name="GitHub Copilot",
        binary=None,
        config_dir=_HOME / ".config" / "github-copilot",
        mcp=McpConfig(_HOME / ".copilot" / "mcp-config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".copilot" / "agents" / "semble-search.agent.md",
    ),
    AgentTarget(
        id="codex",
        display_name="Codex",
        binary="codex",
        config_dir=_HOME / ".codex",
        mcp=McpConfig(_HOME / ".codex" / "config.toml", "mcp_servers", _STDIO_SERVER_CONFIG, format="toml"),
        instructions_path=_HOME / ".codex" / "AGENTS.md",
        subagent_path=_HOME / ".codex" / "agents" / "semble-search.toml",
    ),
    AgentTarget(
        id="zcode",
        display_name="ZCode",
        binary=None,
        config_dir=_HOME / ".zcode",
        mcp=McpConfig(_HOME / ".zcode" / "cli" / "config.json", "mcp.servers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".zcode" / "AGENTS.md",
        subagent_path=_HOME / ".zcode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="vscode",
        display_name="VS Code",
        binary="code",
        config_dir=None,
        mcp=McpConfig(_vscode_mcp_path(), "servers", _STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="windsurf",
        display_name="Windsurf",
        binary="windsurf",
        config_dir=_HOME / ".codeium" / "windsurf",
        mcp=McpConfig(_HOME / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="zed",
        display_name="Zed",
        binary="zed",
        config_dir=_HOME / ".config" / "zed",
        mcp=McpConfig(_HOME / ".config" / "zed" / "settings.json", "context_servers", _ZED_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="reasonix",
        display_name="Reasonix",
        binary="reasonix",
        config_dir=_HOME / ".config" / "reasonix",
        # ~/.reasonix/config.json is the legacy v0.x path still read by v1.x for backwards compat.
        # The v1.x canonical config is ~/.config/reasonix/config.toml ([[plugins]]), but the JSON
        # path requires no special TOML handling and works for new users who have never had v0.x.
        mcp=McpConfig(_HOME / ".reasonix" / "config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "reasonix" / "REASONIX.md",
        subagent_path=_HOME / ".reasonix" / "skills" / "semble-search.md",
    ),
    AgentTarget(
        id="pi",
        display_name="Pi",
        binary="pi",
        config_dir=_HOME / ".pi",
        mcp=McpConfig(_HOME / ".pi" / "agent" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".pi" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="commandcode",
        display_name="Command Code",
        binary=None,
        config_dir=_HOME / ".commandcode",
        mcp=McpConfig(_HOME / ".commandcode" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".commandcode" / "AGENTS.md",
        subagent_path=_HOME / ".commandcode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="antigravity",
        display_name="Antigravity",
        binary="agy",
        config_dir=_HOME / ".gemini" / "antigravity-cli",
        mcp=McpConfig(_HOME / ".gemini" / "config" / "mcp_config.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "config" / "skills" / "semble-search" / "SKILL.md",
    ),
]


def is_detected(agent: AgentTarget) -> bool:
    """Return True if the agent appears to be installed."""
    if agent.binary and shutil.which(agent.binary):
        return True
    return bool(agent.config_dir and _exists_or_denied(agent.config_dir))
