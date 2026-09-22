# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.utils import bind_arguments

ToolFunction = Callable[..., Any]


def _is_primitive(value):
    return isinstance(value, (str, int, bool, float))


def _to_otel_value(python_value):
    """Coerces parameters to something representable with Open Telemetry."""
    if python_value is None or _is_primitive(python_value):
        return python_value
    if isinstance(python_value, list):
        return [_to_otel_value(x) for x in python_value]
    if isinstance(python_value, dict):
        return {
            key: _to_otel_value(val) for (key, val) in python_value.items()
        }
    if hasattr(python_value, "model_dump"):
        return python_value.model_dump()
    if hasattr(python_value, "__dict__"):
        return _to_otel_value(python_value.__dict__)
    return repr(python_value)


def _wrap_tool_function(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
):
    if inspect.iscoroutinefunction(tool_function):

        @functools.wraps(tool_function)
        async def wrapped_function(*args, **kwargs):
            with telemetry_handler.tool(
                tool_function.__name__,
            ) as tool_invocation:
                tool_invocation.tool_description = tool_function.__doc__
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = bind_arguments(
                        tool_function,
                        args,
                        kwargs,
                        apply_defaults=False,
                    )
                result = await tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
            return result
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args, **kwargs):
            with telemetry_handler.tool(
                tool_function.__name__,
            ) as tool_invocation:
                tool_invocation.tool_description = tool_function.__doc__
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = bind_arguments(
                        tool_function,
                        args,
                        kwargs,
                        apply_defaults=False,
                    )
                result = tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
            return result

    return wrapped_function


def wrapped_tool(
    tool_or_tools: ToolFunction
    | ToolOrDict
    | ToolListUnion
    | ToolListUnionDict
    | None,
    telemetry_handler: TelemetryHandler,
):
    if tool_or_tools is None:
        return None
    if isinstance(tool_or_tools, list):
        return [
            wrapped_tool(tool, telemetry_handler) for tool in tool_or_tools
        ]
    if isinstance(tool_or_tools, dict):
        return {
            key: wrapped_tool(tool, telemetry_handler)
            for (key, tool) in tool_or_tools.items()
        }
    if callable(tool_or_tools):
        return _wrap_tool_function(tool_or_tools, telemetry_handler)
    return tool_or_tools
