import requests

def collect_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            return data
        else:
            return None
    except:
        return None

# Example usage
data = collect_data("https://example.com/data")
if data:
    print(data)
else:
    print("Data not collected")