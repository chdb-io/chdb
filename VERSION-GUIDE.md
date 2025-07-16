# chDB Version Numbering Rules

## Version Format

chDB follows the [Semantic Versioning (SemVer)](https://semver.org/) specification, with version format: `vX.Y.ZbN`

### Version Components

- **First digit (1)** - **Major Version**
  - Contains incompatible API changes or major architectural adjustments
  - Requires compatibility checks when upgrading
  - Real examples: `v1.0` → `v2.0` → `v3.0`

- **Second digit (2)** - **Minor Version**
  - New features and backward-compatible API extensions
  - Generally no breaking changes, safe to upgrade
  - Real examples: `v3.0.0` → `v3.1.0` → `v3.2.0` → `v3.3.0` → `v3.4.0`

- **Third digit (3)** - **Patch Version**
  - Backward-compatible bug fixes and minor optimizations
  - Users can upgrade with confidence, no compatibility concerns
  - Real examples: `v3.1.0` → `v3.1.1` → `v3.1.2`

- **Suffix b0** - **Beta Version Identifier**
  - Indicates beta test version, numbering starts from 0
  - Used for feature testing before official release
  - Real examples: `v2.2.0b0` → `v2.2.0b1` → `v2.2.0`

## Beta Version Overview

### What is a Beta Version

Beta versions are pre-release versions before the official release, used for:
- Testing new feature stability
- Collecting user feedback  
- Discovering potential issues

### Beta Version Characteristics

- **pip default installation ignores** beta versions
- Features are relatively stable but may have unknown issues
- Suitable for testing environments, use caution in production

### Installing Beta Versions

```bash
# Install latest beta version
pip install --pre chdb

# Install specific beta version
pip install chdb==2.2.0b0
```

## Real Version Release Examples

### Latest Version Series (v3.x)
- **v3.4.0** (July 2025) - Latest stable version, upgraded ClickHouse to v25.5.2.47
- **v3.3.0** - Added JSON type support and storage metrics
- **v3.2.0** - Added streaming query API
- **v3.1.2** - Fix version, resolved multiple bugs
- **v3.1.1** - Fix version, performance optimizations
- **v3.1.0** - Added JSON type support
- **v3.0.1** - Fix version, resolved v3.0.0 issues
- **v3.0.0** - Major version, introduced connection-based API which is also the default implementation. Users upgrade from v2.x to v3.x should be careful.

### Beta Version Series Examples
- **v2.2.0b0** - First beta version of v2.2.0
- **v2.2.0b1** - Second beta version of v2.2.0, fixed issues found in b0
- **v2.2.0** - Official version release

## Version Upgrade Recommendations

### Safe Upgrades
- **Patch versions** (e.g., v3.1.0 → v3.1.2): Can upgrade directly
- **Minor versions** (e.g., v3.1.x → v3.2.x): Usually safe to upgrade, testing recommended

### Cautious Upgrades
- **Major versions** (e.g., v2.x → v3.x): Need compatibility checks, may require code changes
- **Beta versions**: Only recommended for testing environments

### Production Environment Recommendations
- Use stable versions (no beta suffix)
- Upgrade to latest bugfix version first(eg. v3.1.0 -> v3.1.2)
- Verify compatibility in testing environment before upgrading