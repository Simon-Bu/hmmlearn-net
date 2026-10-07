# Notice

HmmLearn (this repository, `hmmlearn-net`) is a derivative work: a port to .NET 9 of
[hmmlearn](https://github.com/hmmlearn/hmmlearn) 0.3.3 (tag `v0.3.3` plus the changelog commit `e01a10e`), by the
hmmlearn authors and contributors listed in `reference/hmmlearn-0.3.3/AUTHORS.rst`.

- Original work: Copyright (c) 2014, hmmlearn authors and contributors, BSD 3-Clause License. The license text is
  reproduced unchanged in `LICENSE`.
- `reference/hmmlearn-0.3.3/` is an unmodified copy of the parts of the original used as the specification
  (`src/hmmlearn/**`, `ext/_hmmc.cpp`, `LICENSE.txt`, `AUTHORS.rst`). It is read-only.
- `oracle/golden/*.json` holds numbers computed by running the original hmmlearn 0.3.3 on explicit inputs
  (see `oracle/README.md`).
- The C# code in `src/HmmLearn/` is a new implementation written from that specification by AI agents under the
  supervision of the AnythingMachine software factory, as a proof of concept. It is not endorsed by the hmmlearn authors.
