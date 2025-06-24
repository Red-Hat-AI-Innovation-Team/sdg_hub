# Changelog

All notable changes to SDG Hub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Docsify documentation website
- Enhanced web interface with Flask backend
- Improved examples with Jupyter notebooks
- Custom blocks for PDF parsing and data transformation

### Changed
- Refactored SDG Class to use Flows instead of Pipelines
- Enhanced prompt templates with better Jinja2 support
- Improved error handling across all components

### Fixed
- Checkpoint loading and saving reliability
- Block registry initialization issues
- Data validation in utility blocks

## [Latest Release] 

> 📋 **Note**: This changelog will be automatically updated with each release. 
> For the most current release information, see the [GitHub Releases](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases) page.

---

## Release Categories

### 🎉 **Added**
New features and capabilities

### 🔄 **Changed** 
Changes in existing functionality

### 🗑️ **Deprecated**
Soon-to-be removed features

### ❌ **Removed**
Now removed features

### 🐛 **Fixed**
Bug fixes

### 🔒 **Security**
Security improvements

---

## Contributing to Changelog

When contributing to SDG Hub, please update this changelog following these guidelines:

1. **Add entries to the `[Unreleased]` section**
2. **Use the categories above** (Added, Changed, Fixed, etc.)
3. **Write clear, concise descriptions**
4. **Include links to issues/PRs where relevant**
5. **Follow the existing format**

### Example Entry Format

```markdown
### Added
- New `CustomBlock` for advanced data processing ([#123](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/pull/123))
- Support for additional LLM providers in flow configuration

### Fixed  
- Resolved memory leak in large dataset processing ([#124](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/issues/124))
```

---

## Archive

For historical releases and detailed version history, visit:
- [GitHub Releases](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
- [PyPI Release History](https://pypi.org/project/sdg-hub/#history)