# Release policy

LTX Desktop releases fail closed: a public artifact is not published
unless its source tag, automated tests, security checks, dependency evidence,
SBOM, checksums, and any signatures required for the published artifact type
all pass.

## Current distribution status

The project publishes an interim **unsigned installer beta** from a gated,
public GitHub Actions workflow. These packages are official artifacts of this
fork repository, but they do not have Windows Authenticode or Apple
Developer ID signatures and are not Apple-notarized. The download guide
describes the resulting SmartScreen and Gatekeeper warnings.

Local packages remain development artifacts and must not be presented as
official end-user builds. Only assets attached to this repository's Releases
page by the gated workflow are supported.

Future signed Windows and macOS installers require project-owned
Authenticode and Apple Developer credentials plus renewed clean-machine
qualification. Credentials, certificates, and notarization secrets must stay
in platform credential stores or GitHub Actions secrets and must never be
committed.

## Signed release tags

1. Ensure `main` is clean and all required CI and security checks pass.
2. Review release notes, privacy disclosures, screenshots, and tutorial media.
3. Create an annotated, cryptographically signed tag matching `package.json`,
   such as `v1.1.0-beta.2`.
4. Run the applicable release workflow with publication disabled.
5. Download and inspect the evidence bundle. Verify the source archive,
   `sbom.spdx.json`, license reports, `SHA256SUMS`, and GitHub provenance
   attestation.
6. After explicit publication approval, use the workflow's protected publish
   environment.

The workflow rejects lightweight or unverified tags.
`package.json` is the release-version authority. The backend's private Python
package version is internal package metadata and is not independently released.

For an SSH-signed tag, register the public key as a **signing key** on the
maintainer's GitHub account, then use one-command Git options so no repository
or global Git configuration is changed:

```bash
git -c gpg.format=ssh \
  -c user.signingkey="$HOME/.ssh/ltx_release_signing_ed25519" \
  tag -s v1.1.0-beta.2 -m "LTX Desktop v1.1.0-beta.2"
git push origin v1.1.0-beta.2
```

Before dispatching, confirm GitHub reports the annotated tag object as verified.

After downloading the gated evidence bundle, verify it from that directory:

```bash
sha256sum --check SHA256SUMS
gh attestation verify --repo MountainPlatform300/LTX-Desktop ./*
```

## Python runtime artifacts

Windows and Linux download their Python runtime from the matching release in
this repository. The runtime is built from `backend/uv.lock`.

`scripts/generate-python-deps-hash.mjs` derives a platform-specific dependency
identity from the Python version, project metadata, and lockfile.
`scripts/package-python-embed.mjs` creates release-sized parts and a manifest
containing the dependency identity, exact sizes, per-part SHA-256 digests, and
the complete archive SHA-256 digest. The application verifies all of those
values before extraction and rejects path traversal, links, special files,
untrusted production hosts, and integrity mismatches.

The base Python archives are pinned to reviewed SHA-256 digests before
extraction. Registry packages are installed from direct artifact URLs and
SHA-256 digests exported from `uv.lock`; this prevents CUDA/PyPI index
confusion. Git dependencies remain pinned to reviewed commit IDs. The build
does not execute an unversioned `get-pip.py` bootstrap.

CI audits the exported production Python lock with the separately locked
`pip-audit` tool. `GHSA-rrmf-rvhw-rf47` is explicitly excluded because it
affects `torch.jit.script`, which this application does not use; the exception
must be removed when a compatible patched CUDA build is available.

Installer releases must include `python-deps-hash.txt`, and their matching
Python manifest and parts must be uploaded under the same immutable version tag
before installer publication. Source-only releases do not publish these runtime
artifacts.

## Unsigned installer prerelease

The **Installer Release** workflow:

1. checks out an existing annotated, GitHub-verified signed tag whose value
   matches `package.json`;
2. requires successful CI, CodeQL, secret scanning, and dependency gates on
   that commit;
3. builds Windows x64, macOS arm64, and Linux x64 packages on native runners;
4. builds Windows/Linux Python runtimes from the locked dependencies and
   publishes only their verified manifests and numbered parts;
5. confirms Windows and macOS artifacts are intentionally unsigned, rejects
   secret-like packaged files, and inspects Linux package contents;
6. rejects missing, duplicate, unexpectedly large, or misnamed assets;
7. produces source, dependency, SPDX SBOM, SHA-256, and GitHub provenance
   evidence; and
8. stages a draft prerelease before the protected `installer-release`
   environment permits publication.

Automatic installer updates stay disabled during this unsigned phase. Updating
means downloading the next installer manually from the official Releases page.
The existing app-data location remains stable across upgrades.

## Installer qualification

Before enabling installer publication, test each supported package on a clean
machine:

- installation, launch, backend authentication, and health;
- first-run Python and model downloads, including corrupted-download recovery;
- local generation and RunPod training, cancellation, stop, and termination;
- settings and encrypted-credential migration from the previous release;
- offline behavior, manual upgrade from N-1, rollback, uninstall, and data
  retention;
- expected unsigned status, OS-warning instructions, SBOM/checksum
  verification, and GitHub provenance.

Windows x64, macOS arm64, and Ubuntu/Debian x64 must each pass before their
assets are published. A failing platform is removed from the release rather
than documented as supported.
