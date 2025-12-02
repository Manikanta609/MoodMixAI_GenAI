import pandas as pd
import random

class MusicRecommender:
    def __init__(self, songs_csv_path):
        self.songs_df = pd.read_csv(songs_csv_path)
        
    def recommend(self, mood_label, top_k=5):
        """
        Recommends songs based on mood.
        """
        # Filter by mood
        # We need to ensure the CSV mood_tags match our mood labels or we have a mapping
        # Our labels: angry, disgust, fear, happy, sad, surprise, neutral
        # CSV tags in dummy data: happy, sad, angry, neutral, fear
        
        # Simple mapping or direct match
        # Let's assume direct match for now, with some fallbacks
        
        relevant_songs = self.songs_df[self.songs_df['mood_tag'] == mood_label]
        
        if relevant_songs.empty:
            # Fallback: return random songs or specific fallback
            print(f"No songs found for mood: {mood_label}. Returning random selection.")
            return self.songs_df.sample(n=min(top_k, len(self.songs_df))).to_dict('records')
            
        # Sample if we have more than top_k
        if len(relevant_songs) > top_k:
            recommended = relevant_songs.sample(n=top_k)
        else:
            recommended = relevant_songs
            
        return recommended.to_dict('records')

if __name__ == "__main__":
    rec = MusicRecommender("songs.csv")
    print(rec.recommend("happy"))
