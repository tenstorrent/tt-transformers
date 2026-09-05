# Source provenance

The initial standalone package was extracted from `tt-metal` commit
`00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`.

`source_inventory.csv` maps the reviewed source scope to its standalone
destination, or identifies migration-only context archived with the audit tag.
The only implementation subsystem excluded from extraction is MoE.

Detailed migration work logs, generated analyses, and intermediate evidence
are intentionally not part of the active repository. The pre-cleanup audit
tree is preserved by Git tag `tttv2-migration-audit-54648bc`.
