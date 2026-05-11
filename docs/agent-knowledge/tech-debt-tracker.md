# Tech Debt Tracker

Known technical debt organized by priority. Items are removed as they are
resolved; new items are added as they are discovered.

## High Priority

- **Two documentation systems (MkDocs + Next.js) creating drift risk.** Having
  two separate documentation builds means content can diverge. Consolidate to
  a single system or establish a clear ownership boundary between them.

## Medium Priority

- **Some block categories missing in CLAUDE.md block category table.** The
  `blocks/code` category and any newly added categories are not listed in the
  orientation table, making them harder to discover.

## Low Priority

- **mypy config disables import-not-found and import-untyped errors.** This
  suppresses real type-checking issues from untyped or missing dependencies.
  Ideally these should be re-enabled with targeted `type: ignore` comments
  where needed.

- **5 legacy files excluded from mypy checking.** These exclusions accumulate
  untyped code that is invisible to CI. Each should be brought into compliance
  and removed from the exclusion list.
