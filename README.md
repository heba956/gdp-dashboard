# :earth_americas: GDP dashboard template

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
    
## Deploying to Streamlit Community Cloud

1. Commit and push your repository to GitHub (make sure `streamlit_app.py` is at the repository root and `requirements.txt` is present).

2. Visit https://share.streamlit.io and sign in with your GitHub account.

3. Click **New app**, choose the repository and branch (`main`), and set the **Main file** to `streamlit_app.py`.

4. Click **Deploy**. Streamlit will install dependencies from `requirements.txt` and run the app.

Notes:
- If your app requires a sample taxi CSV to display without user upload, add it under `data/taxi.csv` in the repo.
- If you need environment secrets (API keys), add them in the app settings on Streamlit Cloud.
