# Security Policy

## Supported versions

Security fixes are provided for the latest `1.x` release. Older releases and
unreleased development snapshots may be asked to reproduce an issue on the
latest version before a fix is prepared.

## Report a vulnerability

Please do not disclose a suspected vulnerability in a public issue, pull
request, discussion, or social post.

Use GitHub's private vulnerability reporting for FireLens:

<https://github.com/aslanbayli/firelens/security/advisories/new>

If private vulnerability reporting is unavailable, contact the maintainer
through [Ali Aslanbayli's GitHub profile](https://github.com/aslanbayli) to
establish a private reporting channel before sharing technical details.

Include the affected version, operating system, impact, reproduction steps or
proof of concept, and any suggested mitigation. Remove tokens, private source
code, repository contents, and other secrets from the report.

You should receive an acknowledgment within seven days. The maintainer will
work with you to validate the report, agree on a disclosure timeline, and
credit you if desired.

## Deployment notes

FireLens reads local source code and persists source snippets and embeddings in
SQLite. Treat its data directory as sensitive. Configure
`FIRELENS_ALLOWED_ROOTS` as narrowly as possible, keep
`FIRELENS_DATA_DIR` outside indexed repositories, and do not expose the STDIO
MCP process as an unauthenticated network service.
