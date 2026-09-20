"""One validated request per operation, shared by the CLI and the web API.

The web UI makes a promise that this repo cannot afford to break: every panel
shows the command that produces what it displays. A panel that renders a command
string next to numbers computed by a *different* code path would be the same
class of defect as the August 2026 audit's worst finding -- a published figure
with no producing command -- except dressed up to look verified.

So the command is not a caption. It is generated from the same validated request
object that ran, by the same code, and the CLI builds that object too. If a field
cannot be expressed as a CLI flag, it cannot be a field.

That constraint is deliberate and it bites: it means a web-only knob has to earn
a CLI flag before it can exist. ``fit-scorer`` is the cautionary case --
``scorer.fit_scorer`` takes prepared rows, while the CLI handler around it does
tiling, splitting, feature computation and persistence. Mapping a form straight
onto the library function would have reproduced none of that.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class ServiceRequest(BaseModel):
    """A validated, frozen description of one operation.

    Subclasses declare ``command_name`` and ``cli_flags``. Everything else --
    validation, defaults, the rendered command -- falls out of the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The ``satchangegate`` subcommand this request invokes.
    command_name: ClassVar[str] = ""
    #: field name -> CLI flag. A field absent from this map is not renderable,
    #: which is a deliberate error rather than a silently dropped option.
    cli_flags: ClassVar[dict[str, str]] = {}
    #: Boolean fields written as ``--flag/--no-flag`` rather than bare presence.
    paired_bools: ClassVar[frozenset[str]] = frozenset()
    #: Fields that describe *where* rather than *what*. Rendered only when the
    #: caller asks, because a demo command reads better without them.
    location_fields: ClassVar[frozenset[str]] = frozenset({"root", "out"})

    def to_argv(self, *, include_locations: bool = False) -> list[str]:
        """The argv that reproduces this request through the CLI."""
        if not self.command_name:
            raise NotImplementedError(f"{type(self).__name__} declares no command_name")

        argv = [self.command_name]
        for name, value in self.model_dump().items():
            is_location = name in self.location_fields
            if is_location and not include_locations:
                continue
            if value is None:
                continue
            # A location asked for is a location shown, default or not: the
            # reason to ask is to see where a run actually wrote.
            if not is_location and self._is_default(name, value):
                continue
            flag = self.cli_flags.get(name)
            if flag is None:
                raise NotImplementedError(
                    f"{type(self).__name__}.{name} has no CLI flag, so no command can "
                    f"reproduce a request that sets it. Give it a flag or drop the field."
                )
            argv.extend(self._render_option(name, flag, value))
        return argv

    def to_command(self, *, include_locations: bool = False) -> str:
        """The rendered command line, ready to paste."""
        argv = self.to_argv(include_locations=include_locations)
        return "satchangegate " + " ".join(shlex.quote(a) for a in argv)

    def _is_default(self, name: str, value: Any) -> bool:
        field = type(self).model_fields[name]
        factory = field.default_factory
        if field.default is None and factory is None:
            return False
        if factory is None:
            default = field.default
        else:
            # pydantic allows a factory that takes the already-validated data;
            # every factory here is nullary, so fall back rather than guess.
            try:
                default = factory()  # type: ignore[call-arg]
            except TypeError:  # pragma: no cover - no such factory today
                return False
        return bool(_normalise(value) == _normalise(default))

    def _render_option(self, name: str, flag: str, value: Any) -> list[str]:
        if isinstance(value, bool):
            if name in self.paired_bools:
                return [flag if value else _negate(flag)]
            return [flag] if value else []
        if isinstance(value, (list, tuple)):
            out: list[str] = []
            for item in value:
                out.extend([flag, _scalar(item)])
            return out
        return [flag, _scalar(value)]


def _negate(flag: str) -> str:
    """``--vlm`` -> ``--no-vlm``."""
    return f"--no-{flag.removeprefix('--')}"


def _scalar(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalise(value: Any) -> Any:
    """Compare paths and sequences by value, not identity."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(v) for v in value)
    return value
