# GitHub Setup Instructions

This guide walks you through pushing your Network Flow Solver project to GitHub.

## Current Status

✅ Git repository initialized  
✅ Initial commit created (56 files, 7,886 lines)  
✅ .gitignore configured (cache files excluded)  
✅ Ready to push to GitHub

## Step 1: Create GitHub Repository

1. **Go to GitHub:** https://github.com/new

2. **Configure repository:**
   - **Repository name:** `network_flow_solver` (or your preferred name)
   - **Description:** Pure Python network simplex algorithm for minimum-cost flow problems
   - **Visibility:** 
     - ✅ **Public** (recommended - enables free CI/CD, Codecov, showcases your work)
     - Private (if you prefer, but some features require paid plans)
   - **Initialize repository:** 
     - ⚠️ **DO NOT** check "Add a README file"
     - ⚠️ **DO NOT** add .gitignore
     - ⚠️ **DO NOT** choose a license
     - (We already have all of these locally!)

3. **Click "Create repository"**

## Step 2: Connect and Push

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /Users/jeff/experiments/network_flow_solver

# Rename branch to 'main' (modern convention)
git branch -M main

# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/network_flow_solver.git

# Push to GitHub
git push -u origin main
```

**Using SSH instead?** (If you have SSH keys set up):
```bash
git remote add origin git@github.com:YOUR_USERNAME/network_flow_solver.git
git push -u origin main
```

## Step 3: Update URLs in Project Files

After pushing, update these files with your actual GitHub username:

### 3.1 Update `pyproject.toml`

```bash
# Replace 'yourusername' with your GitHub username
sed -i '' 's/yourusername/YOUR_ACTUAL_USERNAME/g' pyproject.toml
```

Or edit manually - search for `yourusername` in:
- `project.urls.Homepage`
- `project.urls.Repository`
- `project.urls.Issues`
- `project.urls.Documentation`

### 3.2 Update `README.md`

```bash
# Replace 'yourusername' in badges
sed -i '' 's/yourusername/YOUR_ACTUAL_USERNAME/g' README.md
```

Or edit manually - update the badge URLs at the top.

### 3.3 Update `.github/workflows/README.md`

```bash
sed -i '' 's/yourusername/YOUR_ACTUAL_USERNAME/g' .github/workflows/README.md
```

### 3.4 Commit the updates

```bash
git add pyproject.toml README.md .github/workflows/README.md
git commit -m "Update repository URLs with actual GitHub username"
git push
```

## Step 4: Configure GitHub Repository Settings

### 4.1 Enable GitHub Actions

1. Go to repository **Settings** → **Actions** → **General**
2. Under "Actions permissions":
   - Select: ✅ **Allow all actions and reusable workflows**
3. Under "Workflow permissions":
   - Select: ✅ **Read and write permissions**
   - Check: ✅ **Allow GitHub Actions to create and approve pull requests**
4. Click **Save**

### 4.2 Set up Branch Protection (Recommended)

1. Go to **Settings** → **Branches**
2. Click **Add rule**
3. Branch name pattern: `main`
4. Configure:
   - ✅ **Require a pull request before merging**
   - ✅ **Require status checks to pass before merging**
     - Search and select: `All Checks Pass`
   - ✅ **Require branches to be up to date before merging**
   - ✅ **Do not allow bypassing the above settings**
5. Click **Create**

### 4.3 Enable Discussions (Optional)

1. Go to **Settings** → **General**
2. Scroll to **Features**
3. Check: ✅ **Discussions**

## Step 5: Optional Integrations

### 5.1 Codecov (Free for Open Source)

1. Visit https://codecov.io
2. Sign in with GitHub
3. Click **Add new repository**
4. Select `network_flow_solver`
5. Copy the upload token
6. In GitHub: **Settings** → **Secrets and variables** → **Actions**
7. Click **New repository secret**
   - Name: `CODECOV_TOKEN`
   - Value: (paste token)
8. Click **Add secret**

**Benefit:** Coverage reports and trending graphs

### 5.2 PyPI Publishing (For Releases)

**When ready to publish to PyPI:**

1. Create account at https://pypi.org
2. Go to **Account Settings** → **API tokens**
3. Click **Add API token**
   - Token name: `GitHub Actions - network_flow_solver`
   - Scope: **Project** → Select your project (after first upload)
4. Copy the token (starts with `pypi-`)
5. In GitHub: **Settings** → **Secrets and variables** → **Actions**
6. Add secret:
   - Name: `PYPI_API_TOKEN`
   - Value: (paste token)

**First PyPI upload** (before automation works):
```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

After first upload, GitHub Actions will auto-publish on releases!

## Step 6: Verify Everything Works

### 6.1 Check GitHub Actions

1. Go to **Actions** tab in your repository
2. You should see the "Initial commit" workflow run
3. Click on it to see all jobs (lint, test, coverage, etc.)
4. All should be green ✅

If any fail:
- Click on the failing job
- Read the error logs
- Common issues:
  - Missing dependencies (should be handled)
  - Platform-specific issues (Windows UMFPACK expected to fail)

### 6.2 View Coverage Report

1. After CI completes, go to **Actions** tab
2. Click on the workflow run
3. Scroll down to **Artifacts**
4. Download `coverage-report`
5. Open `htmlcov/index.html` in browser

### 6.3 Check Badges

Badges in README.md should now show:
- ✅ CI passing
- ✅ Python version
- ✅ License
- 📊 Codecov (after setup)

## Step 7: Make Your First PR

Test the workflow:

```bash
# Create new branch
git checkout -b add-feature

# Make a small change (add a comment somewhere)
echo "# Test comment" >> README.md

# Commit and push
git add README.md
git commit -m "Test: Add comment to README"
git push -u origin add-feature

# Create PR on GitHub
# Watch CI run automatically!
```

## Common Issues & Solutions

### Issue: "Permission denied (publickey)"

**Solution:** Use HTTPS instead of SSH, or set up SSH keys:
```bash
# Remove SSH remote
git remote remove origin

# Add HTTPS remote
git remote add origin https://github.com/YOUR_USERNAME/network_flow_solver.git
git push -u origin main
```

### Issue: CI Fails with "Module not found"

**Solution:** Check that `pyproject.toml` dependencies are correct. The workflow installs with `pip install -e ".[dev,umfpack]"`.

### Issue: Tests pass locally but fail on CI

**Possible causes:**
- Path issues (use absolute imports)
- Missing test fixtures
- Platform differences (Windows vs Linux/Mac)

Check the CI logs for specific errors.

### Issue: Codecov badge shows "unknown"

**Wait:** It can take 5-10 minutes after first push  
**Check:** Ensure `CODECOV_TOKEN` secret is set  
**Verify:** Look at Actions → workflow → coverage job logs

## Next Steps

1. ✅ **Push to GitHub** (follow Step 2)
2. ✅ **Update URLs** (follow Step 3)
3. ✅ **Configure settings** (follow Step 4)
4. ✅ **Set up integrations** (optional, Step 5)
5. ✅ **Verify CI works** (Step 6)
6. 📝 **Share your project!**

## Quick Reference

```bash
# Clone on another machine
git clone https://github.com/YOUR_USERNAME/network_flow_solver.git
cd network_flow_solver

# Set up for development
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,umfpack]"

# Run tests
pytest

# Make changes, commit, push
git add .
git commit -m "Description of changes"
git push
```

## Resources

- [GitHub Docs](https://docs.github.com)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Codecov Documentation](https://docs.codecov.com)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)

---

**Need help?** Open an issue on GitHub or check the documentation!
