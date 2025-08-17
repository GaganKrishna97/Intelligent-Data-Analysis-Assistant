import requests

FIREBASE_URL = "https://your-firebase-app.firebaseio.com/"  # Replace with your Firebase DB URL

def save_workspace(session_id, data):
    url = f"{FIREBASE_URL}/workspaces/{session_id}.json"
    resp = requests.put(url, json=data)
    return resp.ok

def load_workspace(session_id):
    url = f"{FIREBASE_URL}/workspaces/{session_id}.json"
    resp = requests.get(url)
    if resp.ok:
        return resp.json()
    return None
