import re
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY="YOUR_API_KEY"
BASE="https://api.themoviedb.org/3"
GENRES={"action","adventure","animation","comedy","crime","documentary","drama","family","fantasy","history","horror","music","mystery","romance","science fiction","sci-fi","thriller","tv movie","war","western"}

def api(path,**params):
    if API_KEY.startswith("PASTE_"): raise RuntimeError("Add your TMDb v3 API key to API_KEY at the top of main.py.")
    r=requests.get(BASE+path,params={"api_key":API_KEY,**params},timeout=20)
    r.raise_for_status()
    return r.json()

def clean(s):
    return re.sub(r"[^a-z0-9]","",re.sub(r"\(\d{4}\)","",str(s).lower()))

def find_movie(name,movies):
    q=clean(name)
    exact=movies[movies.key==q]
    if not exact.empty: return exact.index[0]
    partial=movies[movies.key.str.contains(q,regex=False)]
    return partial.index[0] if not partial.empty else None

def tmdb_search(name):
    results=api("/search/movie",query=name).get("results",[])
    return results[0] if results else None

def movie_info(title):
    x=tmdb_search(title)
    return (x.get("title",title),x.get("vote_average",0)) if x else (title,"-")

def hybrid(seed,movies,ui,features):
    row=movies.index.get_loc(seed)
    content=cosine_similarity(features[row],features)[0]
    collab=cosine_similarity(ui.loc[[seed]],ui)[0]
    score=.5*content+.5*collab
    score[row]=-1
    return movies.iloc[np.argsort(score)[::-1]].head(5)[["title"]]

def loved_request(q):
    q=re.sub(r"^(i )?(loved|liked|love|like)\s+", "", q,flags=re.I)
    return re.sub(r"^(movies? )?(similar to|like)\s+", "", q,flags=re.I).strip(" .?!")

def genre_request(q):
    q=q.lower().replace("sci fi","science fiction")
    genre=next((g for g in sorted(GENRES,key=len,reverse=True) if re.search(rf"\b{re.escape(g)}\b",q)),None)
    rating=re.search(r"(?:above|over|rating|rated|>=|at least)\s*(\d+(?:\.\d+)?)",q)
    return genre,float(rating.group(1)) if rating else 0

def by_genre(q):
    genre,rating=genre_request(q)
    if not genre: return None
    names=api("/genre/movie/list").get("genres",[])
    gid=next((x["id"] for x in names if x["name"].lower()==genre or (genre=="sci-fi" and x["name"]=="Science Fiction")),None)
    if not gid: return []
    return api("/discover/movie",with_genres=gid,**{"vote_average.gte":rating,"vote_count.gte":500,"sort_by":"vote_average.desc"}).get("results",[])[:5]

def main():
    try:
        movies=pd.read_csv("movies.csv").set_index("movieId")
        ratings=pd.read_csv("ratings.csv")
    except FileNotFoundError:
        print("Put movies.csv and ratings.csv from ml-latest-small beside main.py.")
        return
    movies["key"]=movies.title.map(clean)
    ui=ratings.pivot_table(index="movieId",columns="userId",values="rating").reindex(movies.index).fillna(0)
    features=TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b").fit_transform(movies.genres.str.replace("|"," ",regex=False))
    print("\nAI MOVIE RECOMMENDER\nType 'exit' to close.\nExamples: I loved Interstellar | Recommend horror movies above 8")
    while True:
        q=input("\nYou: ").strip()
        if q.lower() in {"exit","quit","bye"}: break
        try:
            genre,_=genre_request(q)
            results=by_genre(q) if genre else None
            if results is not None:
                if not results: print("No matching movies found.")
                else:
                    print("\nTop recommendations:")
                    for i,x in enumerate(results,1): print(f"{i}. {x['title']}  |  Rating: {x['vote_average']:.1f}")
                continue
            name=loved_request(q)
            remote=tmdb_search(name)
            seed=find_movie(remote["title"] if remote else name,movies)
            if seed is None:
                print("I could not match that movie in MovieLens. Try a well-known movie title.")
                continue
            print(f"\nBecause you liked {movies.at[seed,'title']}:")
            for i,title in enumerate(hybrid(seed,movies,ui,features).title,1):
                title,rating=movie_info(title)
                print(f"{i}. {title}  |  TMDb rating: {rating if rating=='-' else f'{rating:.1f}'}")
        except (requests.RequestException,RuntimeError) as e:
            print(f"\nConnection/API problem: {e}")

if __name__=="__main__": main()
