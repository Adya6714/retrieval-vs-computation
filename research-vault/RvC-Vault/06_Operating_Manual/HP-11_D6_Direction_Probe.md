# HP-11 — Direction-Asymmetry Probe (P1.5)
addresses: [[D06_Direction_Asymmetry]] · phase: 3, inside HP-10's sweep · needs: Fast Downward (in repo toolchain), API budget

PROMPT:
Goal: test rename × direction interaction on matched problem graphs.
Steps:
1. BW: for 30 state graphs, generate assembly (forward) and disassembly (backward) tasks over the identical graph; verify optimal plans both directions with Fast Downward; match BFS step counts across directions per item (control for the search-complexity asymmetry established by arXiv:2411.01790 — cite it; our claim is the interaction, not the asymmetry).
2. SP: reuse W5 source↔destination swaps, 30 items.
3. Grid per item: {forward, backward} × {canonical names, W3 nonce rename} = 4 cells; 5 models, T=0.
4. Analysis: mixed-effects logistic with direction × rename interaction + (1|item). Headline test: does rename shrink the direction gap (templates) or leave it intact (search geometry)?
Output: bank, raw CSVs, D6_REPORT.md with the interaction plot.
Validate: BFS-matching documented per item; any item where directions differ in optimal plan length is excluded and listed.
