# QBES Version Quick Reference

**For:** Users, Developers, and Researchers  
**Updated:** October 23, 2025

---

## What Version Am I Using?

### Check Your Version

```bash
# Method 1: Python
python -c "import qbes; print(qbes.__version__)"

# Method 2: CLI
qbes --version

# Method 3: Check code directly
# Look in: qbes/__init__.py
```

**Current Answer:** `1.1.0`

---

## Version Summary

### One-Line Descriptions

| Version | Description | Status |
|---------|-------------|--------|
| **v1.0** | Initial prototype - 64.7% tests passing | Historical |
| **v1.1** | Stable & usable - 100% tests passing | ✅ **Current** |
| **v1.2** | Validation & robustness (spec) - >98% accuracy target | ⚠️ Partial |

---

## Quick Feature Lookup

**Need to know if a feature is available?**

| Feature | v1.0 | v1.1 | v1.2 Spec | Available? |
|---------|------|------|-----------|------------|
| Basic simulation | ✅ | ✅ | ✅ | ✅ Yes |
| Interactive CLI | ❌ | ✅ | ✅ | ✅ Yes |
| Auto hydrogen fix | ❌ | ✅ | ✅ | ✅ Yes |
| Literature benchmarks | ❌ | ❌ | ✅ | ✅ Yes (in v1.1) |
| Performance profiling | ❌ | ❌ | ✅ | ✅ Yes (in v1.1) |
| Enhanced validation | ❌ | ⚠️ | ✅ | ✅ Yes (in v1.1) |
| Debugging loop | ❌ | ❌ | ✅ | ✅ Yes (in v1.1) |
| `--dry-run` flag | ❌ | ❌ | ✅ | ❌ Not yet |
| `--save-snapshots` | ❌ | ❌ | ✅ | ❌ Not yet |
| `reference_data.json` | ❌ | ❌ | ✅ | ❌ Not yet |

---

## Version Confusion Explained

### Why Does the README Say v1.2?

**Short Answer:** Documentation got ahead of code. The actual code is v1.1.0.

**Long Answer:**
- Code version: `1.1.0` (in `qbes/__init__.py`)
- Documentation version: `1.2.0-dev` (in README, website)
- Reality: v1.1 with some v1.2 features added

**What This Means:**
- You're using a **stable v1.1** with **bonus features**
- Some v1.2 features were implemented early
- Full v1.2 specification not yet complete
- Documentation uses "1.2.0-dev" for marketing/aspirational reasons

### Is This a Problem?

**No,** the software is stable and functional! The version mismatch is just documentation inconsistency.

**For users:** Everything works as documented  
**For developers:** Code version (1.1.0) is source of truth  
**For researchers:** All features described are available

---

## What Should I Cite?

### For Academic Papers

Use the **actual code version** (1.1.0):

```bibtex
@software{qbes2025,
  title={QBES: Quantum Biological Environment Simulator},
  author={Aniket Mehra},
  year={2025},
  version={1.1.0},  ← Use this
  url={https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-}
}
```

### For Informal References

"QBES v1.1" or "QBES version 1.1.0"

---

## Feature Availability by Version

### v1.0 Features (Baseline)

```
✅ Core simulation engine
✅ MD + Quantum coupling
✅ Basic noise models
✅ Simple CLI
✅ Configuration files
✅ Basic analysis
```

### v1.1 Features (Stable Release)

```
✅ All v1.0 features +
✅ Interactive config wizard
✅ 100% test pass rate
✅ Enhanced error messages
✅ Auto hydrogen addition
✅ Comprehensive docs
✅ Troubleshooting guides
```

### v1.1+ Features (Bonus Additions)

These were planned for v1.2 but implemented early:

```
✅ Literature benchmarks
   - FMO Complex (Engel 2007)
   - FMO Complex (Ishizaki 2009)
   - PSII (Romero 2014)
   - Analytical systems

✅ Performance profiling
   - Timing measurements
   - Memory tracking
   - Bottleneck detection
   - Optimization tips

✅ Enhanced validation
   - Density matrix checks
   - Energy conservation
   - Thermalization
   - Coherence decay

✅ Debugging loop
   - Autonomous validation
   - Automatic fixing
   - CHANGELOG generation
```

### v1.2 Spec Features (Not Yet Available)

```
❌ --dry-run flag
❌ --save-snapshots flag
❌ reference_data.json file
❌ >98% accuracy certification
```

---

## Migration Guide

### Coming from v1.0?

**Good news:** Everything is backwards compatible!

**What changed:**
- Tests now pass 100% (was 64.7%)
- New interactive config wizard
- Better error messages
- Auto-fixes for common issues
- Much better documentation

**Action needed:** None, just enjoy the improvements!

### Upgrading Installation

```bash
# Get latest code
cd QBES
git pull origin main

# Reinstall
pip install -e .

# Verify version
python -c "import qbes; print(qbes.__version__)"
# Should show: 1.1.0
```

---

## FAQ

### Q: Why is README version (1.2) different from code version (1.1)?

**A:** Documentation got updated optimistically. Code version is the truth. Think of "1.2.0-dev" as "1.1.0 with development features".

### Q: Which version should I trust?

**A:** Trust the code: `qbes.__version__` = `"1.1.0"`

### Q: Is v1.2 released?

**A:** Not officially. Current release is v1.1.0 with some v1.2 features.

### Q: When will true v1.2 be released?

**A:** When the full v1.2 specification is implemented:
- All planned features complete
- >98% accuracy achieved
- Full self-certification passed

### Q: Can I use this for research?

**A:** Yes! v1.1 is stable and validated:
- 100% test pass rate
- Literature benchmarks included
- Comprehensive validation
- Well documented

### Q: What's the difference between v1.1 and v1.2 spec?

**A:** See `docs/VERSION_HISTORY.md` for detailed comparison.

**Quick summary:**
- v1.1: Stable, usable, 100% tests
- v1.2 spec: Adds validation suite, dry-run, snapshots, >98% accuracy
- Current: v1.1 with some v1.2 features

### Q: Should I wait for v1.2?

**A:** No need! Current version is fully functional and production-ready.

---

## Where to Learn More

### Documentation

- **Full Version History:** `docs/VERSION_HISTORY.md`
- **Complete User Guide:** `docs/technical/complete-user-guide.md`
- **Getting Started:** `COMPLETE_BEGINNERS_GUIDE.md`

### Specification Files

- **v1.1 Spec:** `.kiro qbes/specs/qbes-v1-1-finalization/`
- **v1.2 Spec:** `.kiro qbes/specs/qbes-v1-2-validation-robustness/`

### Code Version Locations

```python
# Primary source of truth
qbes/__init__.py:  __version__ = "1.1.0"

# CLI version
qbes/cli.py:  @click.version_option(version="1.1.0")

# Test expectations
tests/test_validation_summary.py:  assert summary.qbes_version == "1.1.0"
```

---

## Contact

**Questions about versions?**
- Check: `docs/VERSION_HISTORY.md`
- Email: aniketmehra715@gmail.com
- Issues: GitHub repository

---

**Quick Reference Version:** 1.0  
**Last Updated:** October 23, 2025  
**Maintainer:** QBES Development Team
