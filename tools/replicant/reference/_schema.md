# Reference store schema

One file per entry: `reference/<entry>.json`, loaded by `refdata.load`.

Top level is an object mapping key -> record:

```json
{
  "key": {
    "value": "0.7632",
    "tol": "1e-4",
    "cite": "Author Year, Eq. N",
    "independent_route": "how this was cross-checked without the cited derivation",
    "sources_disagree": [{"value": "0.7633", "cite": "Other Year, Table N"}]
  }
}
```

- `value`, `tol`: decimal strings, parsed with mpmath (never floats).
- `cite`: required; values come from the cited paper, never from memory.
  Missing or empty -> `Refusal("unbanked:no-citation")`.
- `independent_route`: required; a second route to the same number.
  Missing or empty -> `Refusal("unbanked:no-independent-route")`.
- `sources_disagree`: optional list of `{"value", "cite"}` for published values
  that differ. `check` accepts `got` within `tol` of the main value or of any
  of these; `tol` must cover the spread.
