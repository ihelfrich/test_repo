# 🚀 Deployment Guide

## Current Status

✅ **All code pushed to GitHub:** https://github.com/ihelfrich/test_repo
⚠️ **GitHub Pages not yet enabled:** Returns 404

## How to Enable GitHub Pages

### Step 1: Access Repository Settings

Go to: https://github.com/ihelfrich/test_repo/settings/pages

### Step 2: Configure Pages

Under **"Build and deployment"** section:

1. **Source:** Select "Deploy from a branch"
2. **Branch:** Select `main`
3. **Folder:** Select `/docs`
4. Click **"Save"**

### Step 3: Wait for Deployment

- Initial deployment takes 1-2 minutes
- GitHub will show a green checkmark when live
- You'll see: "Your site is live at https://ihelfrich.github.io/test_repo/"

### Step 4: Verify Deployment

Once enabled, these URLs will work:

- **Main Visualization:** https://ihelfrich.github.io/test_repo/
- **Landing Page:** https://ihelfrich.github.io/test_repo/landing.html
- **Data API:** https://ihelfrich.github.io/test_repo/data/baci_gravity_viz.json

## What's Included

### Interactive Visualizations

1. **Counterfactual Shock Explorer** (`docs/index.html`)
   - 3D visualization of trade flows
   - Interactive sliders for elasticity parameters
   - Real-time counterfactual calculations
   - Winner/loser leaderboards

2. **Landing Page** (`docs/landing.html`)
   - Professional project showcase
   - Feature highlights
   - Methodology overview
   - Call-to-action buttons

### Data Files

- `docs/data/baci_gravity_viz.json` (434 KB) - Full dataset for web viz
- `docs/data/baci_gravity_viz.parquet` (69 KB) - Columnar format for analysis

### Documentation

- `README.md` - Comprehensive project documentation
- `DEPLOYMENT.md` - This file

## Troubleshooting

### If Pages Returns 404

1. Check that Pages is enabled in Settings → Pages
2. Verify branch is `main` and folder is `/docs`
3. Wait 2-3 minutes after enabling
4. Force refresh browser (Cmd+Shift+R or Ctrl+Shift+R)

### If Visualization Doesn't Load

1. Open browser console (F12)
2. Check for data loading errors
3. Verify `data/baci_gravity_viz.json` exists
4. Check network tab for failed requests

### If You Need to Redeploy

```bash
# Make changes to files in docs/
git add docs/
git commit -m "Update visualization"
git push origin main

# Wait 1-2 minutes for automatic redeployment
```

## Alternative Deployment Options

### Option 1: Root Deployment (Not Recommended)

Instead of `/docs`, you could deploy from root:
- Move `docs/index.html` to root
- Set folder to `/` in Pages settings
- **Downside:** Mixes source code with published site

### Option 2: gh-pages Branch (More Complex)

Create dedicated deployment branch:
```bash
git checkout --orphan gh-pages
git rm -rf .
cp -r docs/* .
git add .
git commit -m "Deploy to gh-pages"
git push origin gh-pages
git checkout main
```

Then set Pages source to `gh-pages` branch.

### Option 3: GitHub Actions (Advanced)

Create `.github/workflows/deploy.yml` for automated builds.

**Recommendation:** Stick with `/docs` folder on `main` branch (current setup).

## Share Your Work

Once live, you can share:

- Direct link: https://ihelfrich.github.io/test_repo/
- Landing page: https://ihelfrich.github.io/test_repo/landing.html
- GitHub repo: https://github.com/ihelfrich/test_repo

Add to:
- LinkedIn projects
- Portfolio website
- Research presentations
- Job applications
- Academic papers (as supplementary material)

## Next Steps After Deployment

1. ✅ Test all interactive features work
2. ✅ Verify data loads correctly
3. ✅ Check mobile responsiveness
4. ✅ Test on different browsers (Chrome, Firefox, Safari)
5. ✅ Share with colleagues for feedback
6. 📈 Consider adding Google Analytics (optional)
7. 🎯 Add Open Graph meta tags for social sharing (optional)

## Need Help?

- GitHub Pages Docs: https://docs.github.com/en/pages
- Three.js Docs: https://threejs.org/docs/
- Project Issues: https://github.com/ihelfrich/test_repo/issues
