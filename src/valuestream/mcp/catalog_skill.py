"""Serve the packaged catalog authoring skill through the MCP Skills extension."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.resources import files
from typing import Any

import yaml
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding, ResourceBinding
from mcp.server.mcpserver.resources.types import TextResource
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, PaginatedRequestParams, RequestParams

SKILL_NAME = "valuestream-catalog-authoring"
SKILL_URI = f"skill://{SKILL_NAME}/SKILL.md"


class SkillLookup(RequestParams):
    uri: str


class CatalogSkill(Extension):
    identifier = "io.modelcontextprotocol/skills"

    def __init__(self) -> None:
        content = files("valuestream.mcp").joinpath("skills", SKILL_NAME, "SKILL.md").read_bytes()
        self.text = content.decode("utf-8")
        frontmatter = yaml.safe_load(self.text.replace("\r\n", "\n").split("---\n", 2)[1])
        self.entry = {
            "uri": SKILL_URI,
            "frontmatter": frontmatter,
            "resources": [
                {
                    "uri": SKILL_URI,
                    "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
                    "size": len(content),
                }
            ],
        }

    def resources(self) -> Sequence[ResourceBinding]:
        return [
            ResourceBinding(
                TextResource(
                    uri=SKILL_URI, name=SKILL_NAME, mime_type="text/markdown", text=self.text
                )
            )
        ]

    def methods(self) -> Sequence[MethodBinding]:
        return [
            MethodBinding("skills/list", PaginatedRequestParams, self.list_skills),
            MethodBinding("skills/get", SkillLookup, self.get_skill),
        ]

    async def list_skills(
        self, ctx: ServerRequestContext, params: PaginatedRequestParams
    ) -> dict[str, Any]:
        if params.cursor is not None:
            raise MCPError(code=INVALID_PARAMS, message="Unknown skill cursor")
        return {
            "resultType": "complete",
            "skills": [self.entry],
            "cacheScope": "public",
            "ttlMs": 300000,
        }

    async def get_skill(self, ctx: ServerRequestContext, params: SkillLookup) -> dict[str, Any]:
        if params.uri != SKILL_URI:
            raise MCPError(code=INVALID_PARAMS, message="Unknown skill URI")
        return {
            "resultType": "complete",
            "skill": self.entry,
            "cacheScope": "public",
            "ttlMs": 300000,
        }
