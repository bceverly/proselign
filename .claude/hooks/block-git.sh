#!/bin/sh
# PreToolUse guard: Claude may never run git or gh in this repository.
# Reads the hook payload on stdin and denies any Bash command that invokes
# git or gh anywhere in the command line (pipelines, subshells, env prefixes,
# absolute paths, xargs, etc.).
cmd=$(jq -r '.tool_input.command // empty')
if printf '%s\n' "$cmd" | grep -Eq '(^|[[:space:];&|(`$/{!])(git|gh)([[:space:];&|)`}]|$)'; then
  jq -n '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny",
    permissionDecisionReason: "BLOCKED: Claude is not allowed to run git or gh commands in this repository. Ask the user to run it themselves."}}'
fi
exit 0
