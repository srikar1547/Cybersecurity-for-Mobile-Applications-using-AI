import requests
import unittest


class FlaskServerTest(unittest.TestCase):
    def test_server_running(self):
        # Define the server URL
        url = "http://127.0.0.1:5000"

        try:
            # Send a GET request to the server
            response = requests.get(url, timeout=5)

            # Check if the status code is 200 (OK)
            self.assertEqual(
                response.status_code, 200, "Server is not running or accessible."
            )

            # Optionally check for specific content in the response
            self.assertIn(
                "Cybersecurity of Mobile Application Using Artificial Intelligence",
                response.text,
                "The server is running but the expected content is not found.",
            )

            print("Server is running and accessible at:", url)

        except requests.exceptions.RequestException as e:
            self.fail(f"Server is not running or accessible. Error: {str(e)}")


if __name__ == "__main__":
    unittest.main()

