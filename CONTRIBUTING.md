# Contributing

Thank you for your interest in contributing to Exemplar Performance.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) when participating in issues, code reviews, and other project discussions.

## Proposing a Change

For a bug, include the workload, hardware, software versions, steps to reproduce, and expected and observed behavior. Remove credentials and sensitive information from logs before sharing them. Report security vulnerabilities using the [security policy](SECURITY.md), rather than opening a public issue.

For a substantial feature or a change to benchmark methodology, open an issue to discuss the approach before starting implementation. Small documentation fixes can go directly to a pull request.

## Preparing Your Change

- Keep each change focused on one problem and explain why it is needed.
- Follow the conventions in the files you edit and update the relevant documentation when behavior changes.
- Use Conventional Commits format: `type: summary` or `type(scope): summary`. Common types include `feat`, `fix`, `docs`, `test`, and `chore`. For example, `docs: clarify installation prerequisites`.
- Run the relevant checks for the component you changed. The [installer README](cli/llmb-install/README.md) and [launcher README](cli/llmb-run/README.md) describe those tools. Workload-specific READMEs describe recipe setup and usage.

## Local CLI Checks

From the repository root, use `uv` to run both CLI test suites with Python 3.12 and the committed lockfiles:

```bash
(cd cli/llmb-install && uv run --locked --python 3.12 --extra dev pytest tests/ ) &&
  (cd cli/llmb-run && uv run --locked --python 3.12 --extra dev pytest tests/ )
```

This runs both public CLI test suites. Workload validation requires the hardware and scheduler described in the recipe README.

## Adding Recipes or Backends

For a new workload, follow the [Recipe Development Guide](cli/llmb-install/docs/recipe_guide.md) and add metadata, a launch script, and a README alongside similar recipes. For a new launcher backend, implement `JobLauncher` and register it in `create_launcher` in [job_launcher.py](cli/llmb-run/src/llmb_run/job_launcher.py). Add mocked submission tests under `cli/llmb-run/tests/` and document its configuration.

## AI-Assisted Contributions

AI-assisted contributions are welcome. Contributors are responsible for reviewing and validating the output for correctness, security, and license compliance.

## Issue Priority

Maintainers prioritize issues based on severity, user impact, and available capacity. To request reprioritization, add affected workloads and impact to the issue.

## Submitting and Reviewing Changes

Open a pull request with a description of the problem, the change, and how you validated it. Link the related issue, if any, and include the exact check commands and their results so reviewers can reproduce your validation. For internal GitLab development, provide the same information in a merge request.

Maintainers review changes for correctness, reproducibility, security, and consistency with the existing recipes and tools. Maintainers aim to provide initial feedback within a week. Follow up in the request if needed. Address feedback in the same request and summarize any new validation after making changes.

## Signing Off Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution
  is your original work, or you have rights to submit it under the same license, or a compatible
  license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your
  changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO (https://developercertificate.org/):

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
