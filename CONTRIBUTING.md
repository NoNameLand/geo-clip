# Contributing to GeoCLIP

Thank you for your interest in contributing to GeoCLIP! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/geo-clip
   cd geo-clip
   ```

2. **Create a development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install development dependencies**
   ```bash
   pip install pytest black flake8 mypy
   ```

## Project Structure

Please familiarize yourself with our [project structure](docs/PROJECT_STRUCTURE.md) before contributing.

- **Core package**: `geoclip/` - Stable, public API
- **Examples**: `examples/` - User-facing examples
- **Experiments**: `experiments/` - Research and experimental code
- **Scripts**: `scripts/` - Utility scripts
- **Tests**: `tests/` - Test suite
- **Documentation**: `docs/` - Documentation files

## Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Python version and dependencies
- Full error traceback

### ✨ Feature Requests

For feature requests, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach
- Examples of similar features in other libraries

### 🔧 Code Contributions

#### Core Package (`geoclip/`)
- Must maintain backward compatibility
- Requires comprehensive tests
- Should follow existing API patterns
- Needs proper documentation

#### Examples (`examples/`)
- Should be self-contained and well-documented
- Must include clear usage instructions
- Should demonstrate best practices

#### Experiments (`experiments/`)
- Research-focused code
- Less strict requirements than core package
- Should include brief documentation of purpose

## Coding Standards

### Code Style
- Use Black for formatting: `black .`
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)

### Naming Conventions
- Functions and variables: `snake_case`
- Classes: `PascalCase`  
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### Documentation
- Use Google-style docstrings
- Include examples in docstrings for public APIs
- Update relevant documentation files

### Testing
- Add tests for new functionality
- Maintain or improve code coverage
- Run tests before submitting: `pytest tests/`

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black --check .
   flake8 .
   
   # Type checking (optional but recommended)
   mypy geoclip/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test-related changes
   - `refactor:` for code refactoring

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Pull request checklist**
   - [ ] Tests pass
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Backward compatibility maintained (for core package)
   - [ ] Clear description of changes

## Review Process

- Core maintainers will review pull requests
- Feedback will be provided for improvements
- Once approved, maintainers will merge the PR

## Questions and Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact maintainers directly for sensitive issues

## Code of Conduct

Please be respectful and inclusive in all interactions. We aim to maintain a welcoming environment for all contributors.

## Recognition

Contributors will be acknowledged in:
- Git commit history
- Release notes for significant contributions
- README acknowledgments section

Thank you for helping make GeoCLIP better! 🌍
