# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Fraud detection advanced example
- Video tutorial series
- Blog post on Medium/Dev.to
- Community showcase section

## [1.0.0] - 2025-01-18

### Added - Initial Release 🎉

#### Core Examples
- **Example 01**: Medical Expert System
  - Diagnosis with logical rules
  - Uncertainty handling with bounds
  - Bidirectional inference
- **Example 02**: Movie Recommendation System
  - Preference inference from viewing history
  - Similarity-based recommendations
  - Learning from user behavior
- **Example 03**: Hybrid LNN + LLM System
  - Integration with Anthropic Claude
  - Fact extraction from natural language
  - Structured reasoning pipeline
- **Example 04**: Learning with LNN
  - Weight learning from data
  - Exception handling in rules
  - Training with logical contradictions

#### Documentation
- Complete theory guide (`docs/teoria_lnn.md`)
- Installation instructions (`docs/installation.md`)
- Comprehensive comparisons with other approaches (`docs/comparisons.md`)
- Colab setup guide (`docs/colab_setup.md`)
- Real World Applications section in README

#### Infrastructure
- Full test suite with pytest
- GitHub Actions for CI/CD
- Issue and PR templates
- Contributing guidelines
- Code of Conduct
- MIT License

#### Jupyter Notebooks
- Interactive Colab-ready notebooks for all examples
- One-click launch badges in README
- DRY architecture (notebooks import from .py modules)

### Technical Details
- Python 3.9+ support
- IBM LNN framework integration
- PyTorch for training
- Anthropic API integration (Example 03)
- ~80% test coverage
- Black code formatting
- Flake8 linting

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this changelog.

---

[Unreleased]: https://github.com/YOUR_USERNAME/neuro_llm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/YOUR_USERNAME/neuro_llm/releases/tag/v1.0.0
