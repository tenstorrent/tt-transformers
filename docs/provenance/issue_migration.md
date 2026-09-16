# Issue migration

The `tt_transformer v2.1.0` milestone was recreated in this repository from
`tt-metal` [milestone 40](https://github.com/tenstorrent/tt-metal/milestone/40)
as local [milestone 1](https://github.com/tenstorrent/tt_transformers/milestone/1),
carrying the original title and description.

Issues were recreated rather than transferred with `gh issue transfer`.
Transfer would have moved the originals out of `tt-metal`, and because this
repository is internal while `tt-metal` is public, the old public URLs would
have redirected into a repository external readers cannot access. Recreating
leaves the `tt-metal` history intact and publicly readable.

## Scope

The 27 open issues were migrated. The 9 closed issues were deliberately left
in `tt-metal`: the work is finished, and recreating them here would add
issues that are born closed and carry no further work. They remain the
historical record at their original URLs.

## Transforms applied

Each recreated issue body is the original body with three changes:

- Bare `#NNNN` references were expanded to absolute `tt-metal` URLs so they
  keep resolving against the source repository. Left as written they would
  have re-resolved against this repository and gone inert. References already
  written as full markdown links were left untouched. One malformed link in
  the notes carried onto [#27](https://github.com/tenstorrent/tt_transformers/issues/27),
  whose href was the same-page anchor `#55977`, was repaired to the real PR
  URL; it was already broken in `tt-metal`.
- Comments worth keeping were folded into the body as quoted blocks under a
  `Notes carried over from tt-metal` heading, attributed to their original
  author and date. They were not reposted as real comments, which would have
  attributed one person's words to the account running the migration and
  stamped them with the migration date. Four issues carried notes:
  [#9](https://github.com/tenstorrent/tt_transformers/issues/9),
  [#14](https://github.com/tenstorrent/tt_transformers/issues/14),
  [#8](https://github.com/tenstorrent/tt_transformers/issues/8) and
  [#27](https://github.com/tenstorrent/tt_transformers/issues/27).
  The remaining comments were on closed issues and were bookkeeping only.
- A `Migrated from` footer links each issue back to its `tt-metal` original.

Labels were recreated by name and colour before migration: `TT-Transformers`,
`models` and `feature`. All 27 issues are assigned to the migration owner
rather than preserving the original assignees.

## Pending: back-pointers on the tt-metal originals

The `tt-metal` originals have **not** been annotated or closed. Each of the 27
still sits open in `tt-metal` with no indication that work has moved here.

This was deferred on purpose. Posting 27 comments on a public repository
pointing at an internal one gives external readers a dead end, so the
back-pointers should be written once this repository is public.

When that happens, for each row in the mapping below:

1. Comment on the `tt-metal` issue pointing at its counterpart here.
2. Close the `tt-metal` issue as not planned, so the milestone drains.

Until then, treat the mapping table as the authoritative link between the two
repositories.

## Mapping

| tt-metal | tt_transformers | Title |
| --- | --- | --- |
| [#33790](https://github.com/tenstorrent/tt-metal/issues/33790) | [#9](https://github.com/tenstorrent/tt_transformers/issues/9) | Huggingface adaptors |
| [#34286](https://github.com/tenstorrent/tt-metal/issues/34286) | [#10](https://github.com/tenstorrent/tt_transformers/issues/10) | add 2D test cases to `test_distribute_as.py` |
| [#34288](https://github.com/tenstorrent/tt-metal/issues/34288) | [#11](https://github.com/tenstorrent/tt_transformers/issues/11) | Add LazyStateDict |
| [#34898](https://github.com/tenstorrent/tt-metal/issues/34898) | [#12](https://github.com/tenstorrent/tt_transformers/issues/12) | Add support for BH devices for 1D and 2D modules and models |
| [#34921](https://github.com/tenstorrent/tt-metal/issues/34921) | [#13](https://github.com/tenstorrent/tt_transformers/issues/13) | Fingerprinting module-level unit test |
| [#35088](https://github.com/tenstorrent/tt-metal/issues/35088) | [#14](https://github.com/tenstorrent/tt_transformers/issues/14) | Add serialization methods |
| [#35089](https://github.com/tenstorrent/tt-metal/issues/35089) | [#8](https://github.com/tenstorrent/tt_transformers/issues/8) | TTTv2 module serialization |
| [#35500](https://github.com/tenstorrent/tt-metal/issues/35500) | [#15](https://github.com/tenstorrent/tt_transformers/issues/15) | Refactor code for default config among all the modules and tests |
| [#35546](https://github.com/tenstorrent/tt-metal/issues/35546) | [#16](https://github.com/tenstorrent/tt_transformers/issues/16) | Improve unit testing speed |
| [#36158](https://github.com/tenstorrent/tt-metal/issues/36158) | [#17](https://github.com/tenstorrent/tt_transformers/issues/17) | Module-level traced, performance testing |
| [#36293](https://github.com/tenstorrent/tt-metal/issues/36293) | [#18](https://github.com/tenstorrent/tt_transformers/issues/18) | make the unit test files friendlier for AI agents |
| [#37162](https://github.com/tenstorrent/tt-metal/issues/37162) | [#19](https://github.com/tenstorrent/tt_transformers/issues/19) | Sort through any remaining TODOs in the TTTv2 code |
| [#37196](https://github.com/tenstorrent/tt-metal/issues/37196) | [#20](https://github.com/tenstorrent/tt_transformers/issues/20) | Implement TTTv1 models with TTTv2 modules |
| [#37198](https://github.com/tenstorrent/tt-metal/issues/37198) | [#21](https://github.com/tenstorrent/tt_transformers/issues/21) | Meet stakeholders to discuss freezing TTTv1 development |
| [#37200](https://github.com/tenstorrent/tt-metal/issues/37200) | [#22](https://github.com/tenstorrent/tt_transformers/issues/22) | Add config fields to modules |
| [#37201](https://github.com/tenstorrent/tt-metal/issues/37201) | [#23](https://github.com/tenstorrent/tt_transformers/issues/23) | refactor the default configs of all modules |
| [#37256](https://github.com/tenstorrent/tt-metal/issues/37256) | [#24](https://github.com/tenstorrent/tt_transformers/issues/24) | Improve TTTv2 modules by absorbing code/ideas from llama-70b-galaxy work |
| [#37669](https://github.com/tenstorrent/tt-metal/issues/37669) | [#25](https://github.com/tenstorrent/tt_transformers/issues/25) | Prefetcher module |
| [#37820](https://github.com/tenstorrent/tt-metal/issues/37820) | [#26](https://github.com/tenstorrent/tt_transformers/issues/26) | Remove TTTv1 imports in TTTv2 modules |
| [#37855](https://github.com/tenstorrent/tt-metal/issues/37855) | [#27](https://github.com/tenstorrent/tt_transformers/issues/27) | Start Freezing TTTv1 development |
| [#44223](https://github.com/tenstorrent/tt-metal/issues/44223) | [#28](https://github.com/tenstorrent/tt_transformers/issues/28) | comb through every ttnn op that TTT depends on to: |
| [#46500](https://github.com/tenstorrent/tt-metal/issues/46500) | [#29](https://github.com/tenstorrent/tt_transformers/issues/29) | speculative decode |
| [#46502](https://github.com/tenstorrent/tt-metal/issues/46502) | [#30](https://github.com/tenstorrent/tt_transformers/issues/30) | TTTv2 module bringup |
| [#50257](https://github.com/tenstorrent/tt-metal/issues/50257) | [#31](https://github.com/tenstorrent/tt_transformers/issues/31) | [models]: TTTv2 proper KV-cache capacity validation |
| [#51374](https://github.com/tenstorrent/tt-metal/issues/51374) | [#32](https://github.com/tenstorrent/tt_transformers/issues/32) | [TTTv2] Add native HF-layout RoPE support for Llama 3.1 8B |
| [#54000](https://github.com/tenstorrent/tt-metal/issues/54000) | [#33](https://github.com/tenstorrent/tt_transformers/issues/33) | [TTTv2] Define stable public API contracts for model runtime components |
| [#56122](https://github.com/tenstorrent/tt-metal/issues/56122) | [#34](https://github.com/tenstorrent/tt_transformers/issues/34) | Move TTTv2 unit tests from tt-metal to tt_transformers |
