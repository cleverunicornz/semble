"""Pinned sources and identities for the full Rust contrastive experiment."""

from __future__ import annotations

from dataclasses import dataclass

RUST_DATASET_ID = "Fortytwo-Network/Strandset-Rust-v1"
RUST_DATASET_REVISION = "0a8d223302712a2b34a6ad4ce1fd679031894b3d"
RUST_TRAIN_ROWS = 30_893
RUST_EVAL_ROWS = 3_750
RUST_TRAIN_SHA256 = "80d925621777cc890727a4bfe09fb39f12458e183c56b948c8238bf2f755a9c0"
RUST_EVAL_SHA256 = "7c68fc4086008a989dc86984786ce6330a54dc46575de25b2c37d1e5ab7a9966"
RUST_MANIFEST_SHA256 = "f914690f7f4ebd1f2382ced53c8769b75e9a269645d25aed960d114f2d24ba19"

POTION_V1_ID = "minishlab/potion-code-16M"
POTION_V1_REVISION = "1b0ff71095656b23306542bbad34a09109673720"
POTION_V2_ID = "minishlab/potion-code-16M-v2"
POTION_V2_REVISION = "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b"


@dataclass(frozen=True, slots=True)
class ReplaySource:
    """One immutable CornStack language source."""

    language: str
    dataset: str
    revision: str


REPLAY_SOURCES: tuple[ReplaySource, ...] = (
    ReplaySource("go", "nomic-ai/cornstack-go-v1", "570a654bad26af77c7a9f8d719ff3549105da5f9"),
    ReplaySource("java", "nomic-ai/cornstack-java-v1", "3b34339f3bbdcc7b007f871d743f2c146a057364"),
    ReplaySource(
        "javascript",
        "nomic-ai/cornstack-javascript-v1",
        "179dba47cff12683ca12d8f6e4c7c37dff0d4d77",
    ),
    ReplaySource("php", "nomic-ai/cornstack-php-v1", "2a026ddb5edbd8b400ed418c91ad4f6b2c0517ee"),
    ReplaySource("python", "nomic-ai/cornstack-python-v1", "25fb04bd3537983a622d01104a967a5a7f9eaef8"),
    ReplaySource("ruby", "nomic-ai/cornstack-ruby-v1", "292224632eef89f93a337e85a09be564c4d2d1ce"),
)

REPLAY_SEED = 42
REPLAY_BUFFER_SIZE = 10_000
MIN_TEXT_CHARACTERS = 32
