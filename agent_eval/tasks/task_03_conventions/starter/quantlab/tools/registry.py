"""Allowlisted tool registry."""

from collections.abc import Callable

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        class DummyArgs(BaseModel):
            query: str

        def dummy_tool(args: DummyArgs):
            return {"status": "ok"}

        self.register("inspect_state", "Inspects current workflow state", DummyArgs, dummy_tool)

    def register(self, name: str, description: str, args_schema: type[BaseModel], func: Callable):
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            args_schema=args_schema,
            func=func,
        )

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
