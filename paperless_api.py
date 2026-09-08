import os
import requests
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PAPERLESS_API_URL = os.getenv("PAPERLESS_API_URL")
PAPERLESS_API_TOKEN = os.getenv("PAPERLESS_API_TOKEN")

document_types = {}
tags = {}

def fetch_data() -> Dict:
    """Fetch documents from Paperless-ngx API."""
    print(f"Fetching documents from {PAPERLESS_API_URL}...")
    headers = {
        "Authorization": f"Token {PAPERLESS_API_TOKEN}",
        "Accept": "application/json"
    }
    try:
        # Using verify=False to allow self-signed certificates
        response = requests.get(f"{PAPERLESS_API_URL}/api/documents/?page_size=100&fields=title%2Cid%2Ccontent%2Ccreated%2Cmodified%2Ctags%2Cdocument_type", headers=headers, timeout=10, verify=False)
        response.raise_for_status()

        try:
            global document_types
            document_types = fetch_document_types()
        except Exception as e:
            print(f"Failed to fetch latest document types. Using old mappings., {e}")

        try:
            global tags
            tags = fetch_tags()
        except Exception as e:
            print(f"Failed to fetch latest tags. Using old mappings, {e}")

        message = response.json()

        # Replace IDs with names
        for result in message["results"]:
            tag_ids = result["tags"]
            result["tags"] = [tags.get(tag_id, "Unknown Tag") for tag_id in tag_ids]

            doc_type_id = result["document_type"]
            result["document_type"] = document_types.get(doc_type_id, "Unknown Document Type")

        return message
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return {}

def fetch_document_types() -> Dict:
    """Fetch document type names from Paperless-ngx API and return a dict of ID:name pairs."""
    headers = {
        "Authorization": f"Token {PAPERLESS_API_TOKEN}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(f"{PAPERLESS_API_URL}/api/document_types/", headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        return {item["id"]:item["name"] for item in response.json().get("results", [])}
    except Exception as e:
        print(f"Error fetching document types: {e}")
        return {}

def fetch_tags() -> Dict:
    """Fetch tag names from Paperless-ngx API and return a dict of ID:name pairs."""
    headers = {
        "Authorization": f"Token {PAPERLESS_API_TOKEN}",
        "Accept": "application/json"
    }

    try:
        response = requests.get(f"{PAPERLESS_API_URL}/api/tags/", headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        return {item["id"]:item["name"] for item in response.json().get("results", [])}
    except Exception as e:
        print(f"Error fetching tag names: {e}")
        return {}

def fetch_document_content(doc_id: int) -> str:
    """Fetch content of a specific document."""
    headers = {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}
    try:
        # Using verify=False because the user is using a self-signed certificate on their local network
        response = requests.get(f"{PAPERLESS_API_URL}/api/documents/{doc_id}/content/", headers=headers, timeout=10, verify=False)
        
        print(f"DEBUG: Request URL: {response.url}")
        print(f"DEBUG: Status Code: {response.status_code}")
        print(f"DEBUG: Request Headers: {headers}")
        
        response.raise_for_status()
        
        content = response.text
        print(f"--- Raw content snippet for document {doc_id}: ---")
        print(content[:500])
        print("-" * 40)
        return content
    except Exception as e:
        print(f"Error fetching content for document {doc_id}: {e}")
        return ""

def get_tags():
    """Returns a dict mapping of IDs and tag names."""
    return tags

def get_doc_types():
    """Returns a dict mapping of IDs and document type names."""
    return document_types
