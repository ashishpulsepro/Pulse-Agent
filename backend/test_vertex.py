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


# #
# Run this on terminal      and       Billing set karo bas , it will start working 

# aim@aim-Latitude-5410:~$ export GOOGLE_APPLICATION_CREDENTIALS="/home/aim/.gcp_keys/gen-lang-client-0991170889-3f42dde77ebc.json"(1st command)

# aim@aim-Latitude-5410:~$ echo $GOOGLE_APPLICATION_CREDENTIALS(2nd command)
# /home/aim/.gcp_keys/gen-lang-client-0991170889-3f42dde77ebc.json

# aim@aim-Latitude-5410:~$ gcloud auth application-default print-access-token(3rd command)
# ya29.c.c0ASRK0GYIAQxDc5SHBZzpwkWZNuxOT8itRTeWLIHo6J_Krf3bapHvI1jbpLrKIMBpseuLioQLoBKFV10m50eXb8syJV_W8G_lWF-OQOJFgadRL15EqHq1KcWzIy3WF4almZhSrhJ2C-2sxW89qWfd9FhmD-iY4hIpWrCTer5VEZjGCSqxMOZ6yo46plG8e63sbpgxLdUu_fSfMojn1OZ6pyKFO_T8Ec1P1rtBwH0i3qaIUsdPxjfrxhyNd4XLwbswlGvMrVxwJoQHqrXiTD3KxA4jf4TNtUsYKFlwX15HHQArZUIUeEDgOptSnFeZcBY3Rgx1Vlxy9pOJr5XQuu1inpsjbVNKRv211HCfQZBTcydClIcOAvx3-74QggG387KMs9iRobWJsYjW9RRlidp0_4zkl7mgIFwy9Vnh8x8eoVZr6XokRh34aUu3zvtw-nbxBJJV4qkgZjrV4nF6qkwVjV1kt91gbwkzvZYw_ofJ50lidmyZ0WkVtyvXn8eWh5Ofk3cX_s8q0-b5q94hRd5QJIB908djzobeQxXcVuRoZU3ieV5Xx-pnMWU5b-hohgxWWzVVr0pjatMVwhi6J-j-ZnOZta3uhi6SOy2U0Y-S6MnzX7Fm19kUjzp61rrQO9vbWOQyfaZX09F1_crUndhI7gFfq7RnlwbbB3RW_U4Feyn11woe59BaccFVZXf-1xYhoZagb29f-5kgkrIn6zlpYRQ8sQY7tbh2ZJiV4uiUhcIZc9dVv9c8j_ZhUx85a2-9xhq__lSkMxX_zimg-QZndaq3g2QiqRbyg0Y_Q_gqbpUXomMr7Fs53wlB7-2JbiMe_a2OSk9mvlxZwewnzVZ67ob9nnWBuWgcdm_Y87c-pZfnFWfgtn0m4YO5zmoBtVWIU5urmxS-JspgY63m6Q5fhpqxeh9Uhic2f3U5BQdp6v4f0f_kxZzQospcFFMbqRUYRg60FdRW3o-29bOFgpUJtWqI5rpjeMdUZm0Zzyzydha1R1XYuyVvsWn
