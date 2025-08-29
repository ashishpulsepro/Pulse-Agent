from vertexai.generative_models import GenerativeModel
import vertexai
from google.oauth2 import service_account

# Path to your service account JSON
key_path = "/home/aim/.gcp_keys/gen-lang-client-0991170889-3f42dde77ebc.json"

# Load credentials
credentials = service_account.Credentials.from_service_account_file(key_path)

# Initialize Vertex AI
vertexai.init(
    project="gen-lang-client-0991170889",   # ✅ your project ID
    location="us-central1",                 # region you enabled Vertex AI in
    credentials=credentials
)

# Load the Gemini 2.5 Flash Lite model
model = GenerativeModel("gemini-2.5-flash-lite")

# Send a test prompt
response = model.generate_content("Write a short poem about the sky.")

print("✅ Gemini response:")
print(response.text)
