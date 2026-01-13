# Interactive Trade Visualization Dashboard

This directory contains standalone HTML visualizations of the gravity model analysis that can be deployed to GitHub Pages.

## Visualizations

1. **trade_globe_3d.html** - Interactive 3D globe showing top 100 bilateral trade flows as arcs
2. **trade_heatmap.html** - Bilateral trade intensity heatmap (top 30 exporters × importers)
3. **coefficient_plot.html** - PPML gravity model coefficients with 95% confidence intervals
4. **trade_time_series.html** - Animated bar chart of top exporters over time
5. **distribution_plots.html** - Four-panel distribution analysis
6. **index.html** - Main dashboard page linking all visualizations

## How to View Locally

Simply open `index.html` in any web browser. All visualizations are self-contained and work offline.

## How to Deploy to GitHub Pages

### Option 1: Deploy from this outputs/dashboard folder

1. Push your repo to GitHub
2. Go to Settings → Pages
3. Set source to "Deploy from a branch"
4. Select branch `main` and folder `/outputs/dashboard`
5. Save and wait for deployment
6. Your dashboard will be at: `https://[username].github.io/[repo-name]/`

### Option 2: Copy to docs/ folder (GitHub Pages default)

```bash
# From project root
mkdir -p docs
cp -r outputs/dashboard/* docs/
git add docs/
git commit -m "Add interactive dashboard"
git push
```

Then in GitHub Settings → Pages, set source to `/docs` folder.

### Option 3: Copy to root for gh-pages branch

```bash
# Create orphan gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
cp -r outputs/dashboard/* .
git add .
git commit -m "Initial dashboard deployment"
git push origin gh-pages
git checkout main
```

Then in GitHub Settings → Pages, set source to `gh-pages` branch.

## Customization

To regenerate visualizations with different parameters:

```bash
# From project root
source baileym_test/bin/activate
python scripts/04_interactive_dashboard.py --help
```

## Technologies Used

- **Plotly.js** - Interactive JavaScript charting library
- **Plotly Python** - For generating visualizations from pandas DataFrames
- Pure HTML/CSS/JS - No server required, works entirely client-side

## File Sizes

All HTML files combined are typically < 10MB, well within GitHub's limits.

## Browser Compatibility

Works in all modern browsers:
- Chrome/Edge (recommended for best performance)
- Firefox
- Safari
- Mobile browsers (iOS Safari, Chrome Mobile)

## Notes

- Visualizations are fully interactive: zoom, pan, hover for details, filter
- No data leaves your browser - all processing is client-side
- Works offline after initial load
- Responsive design adapts to mobile/tablet screens
