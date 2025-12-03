import pandas as pd
import urllib.request
import urllib.error

def check_links():
    try:
        df = pd.read_csv('songs.csv')
        print(f"Checking {len(df)} songs...")
        
        for index, row in df.iterrows():
            url = row['url']
            title = row['title']
            try:
                # Just check if we can open the URL (basic availability)
                # We'll use a User-Agent to avoid immediate 403s from YouTube
                req = urllib.request.Request(
                    url, 
                    data=None, 
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                )
                with urllib.request.urlopen(req) as response:
                    if response.getcode() == 200:
                        # Read a bit to see if it's "Video unavailable"
                        content = response.read(100000).decode('utf-8', errors='ignore')
                        if "Video unavailable" in content:
                             print(f"❌ {title}: Video unavailable (Page loaded but content restricted)")
                        else:
                             print(f"✅ {title}: OK")
                    else:
                        print(f"❌ {title}: Status {response.getcode()}")
            except urllib.error.HTTPError as e:
                print(f"❌ {title}: HTTP Error {e.code}")
            except Exception as e:
                print(f"❌ {title}: Error {e}")
                
    except Exception as e:
        print(f"Failed to read CSV: {e}")

if __name__ == "__main__":
    check_links()
