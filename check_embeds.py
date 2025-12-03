import pandas as pd
import requests
import json
import time

def check_embeddability():
    df = pd.read_csv('songs.csv')
    print(f"Checking embeddability for {len(df)} songs...")
    
    problematic_songs = []
    
    for index, row in df.iterrows():
        url = row['url']
        title = row['title']
        
        # YouTube oEmbed endpoint
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        
        try:
            response = requests.get(oembed_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {title}: Embeddable")
            elif response.status_code == 401:
                print(f"❌ {title}: Restricted (401)")
                problematic_songs.append(title)
            elif response.status_code == 403:
                print(f"❌ {title}: Restricted (403)")
                problematic_songs.append(title)
            elif response.status_code == 404:
                print(f"❌ {title}: Not Found (404)")
                problematic_songs.append(title)
            else:
                print(f"⚠️ {title}: Status {response.status_code}")
                # problematic_songs.append(title) # Maybe?
                
        except Exception as e:
            print(f"⚠️ {title}: Error {e}")
            
        time.sleep(0.2) # Be nice to the API

    print("\nProblematic Songs to Remove:")
    for song in problematic_songs:
        print(f"- {song}")

if __name__ == "__main__":
    check_embeddability()
