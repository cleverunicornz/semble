from semble.git_workspace import (
    GitChangeClassifier,
    GitWorkspaceSession,
    fetch_remote_baseline,
    open_git_workspace,
    resolve_revision,
)
from semble.index import SembleIndex
from semble.types import Chunk, ContentType, EmbeddingMatrix, IndexStats, SearchResult
from semble.version import __version__
from semble.workspace import (
    BaselineIdentity,
    BaselineRegistry,
    ChangeKind,
    SearchOrigin,
    SearchScope,
    WorkspaceFileChange,
    WorkspaceIndex,
    WorkspaceSearchHit,
    WorkspaceSearchResponse,
    WorkspaceWatcher,
)

__all__ = [
    "BaselineIdentity",
    "BaselineRegistry",
    "ChangeKind",
    "GitChangeClassifier",
    "GitWorkspaceSession",
    "Chunk",
    "ContentType",
    "EmbeddingMatrix",
    "IndexStats",
    "SearchResult",
    "SearchOrigin",
    "SearchScope",
    "SembleIndex",
    "fetch_remote_baseline",
    "resolve_revision",
    "open_git_workspace",
    "WorkspaceFileChange",
    "WorkspaceIndex",
    "WorkspaceSearchHit",
    "WorkspaceSearchResponse",
    "WorkspaceWatcher",
    "__version__",
]
