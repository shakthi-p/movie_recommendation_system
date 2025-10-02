# Movie Recommendation System
A **Netflix-style Movie Recommendation Web App** built with Python and Flask. Get movie suggestions based on similarity and view professional detail pages for each movie.  

## Tech Stack
- **Frontend:** HTML, CSS (Netflix-style styling), Jinja2 templates  
- **Backend:** Python, Flask  
- **Data:** CSV dataset with movies, genres, overview, keywords  
- **Machine Learning:** TF-IDF vectorizer, Cosine Similarity  

## Features
- Search for a movie and get **top similar recommendations**  
- Click a movie to see **detailed information** (overview, genres, release date, tagline, language, popularity)  
- Clean **Netflix-style UI**  
- Fully responsive grid layout  

## How It Works
1. **User enters a movie name** on the homepage  
2. The backend uses **TF-IDF** on combined features (overview + genres + keywords)  
3. **Cosine similarity** is computed to find similar movies  
4. Top N similar movies are displayed in a **Netflix-style card grid**  
5. Clicking a movie opens its **detail page**  

## Run Locally
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate  

Install dependencies:
```bash
pip install -r requirements.txt

Run the app:
```bash
python app.py

Open your browser and visit:
http://127.0.0.1:5000

## Expected Output
Homepage with **search bar** and **suggested movie cards**
<img width="1919" height="992" alt="home_Page" src="https://github.com/user-attachments/assets/4123e037-c83f-469f-9246-3449947c0389" />
Movie **recommendation grid** after search
**Detail page** with complete info for the selected movie


