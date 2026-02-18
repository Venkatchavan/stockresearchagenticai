"""
Simple test to verify Mistral API connectivity and configuration
"""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

# Configure LiteLLM
litellm.drop_params = True
litellm.num_retries = 3
litellm.request_timeout = 60

def test_mistral_connection():
    """Test basic Mistral API connection with a simple request."""
    
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found in environment variables")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print(f"✓ Model: mistral/mistral-small-latest")
    print("\n🔄 Testing API connection with simple request...\n")
    
    try:
        # Simple test request
        response = litellm.completion(
            model="mistral/mistral-small-latest",
            messages=[
                {"role": "user", "content": "Say 'API connection successful' in exactly 3 words."}
            ],
            api_key=api_key,
            timeout=60,
            num_retries=3
        )
        
        result = response.choices[0].message.content
        print("✅ SUCCESS! Mistral API is responding correctly.")
        print(f"Response: {result}")
        print(f"\nToken usage: {response.usage}")
        return True
        
    except litellm.exceptions.AuthenticationError as e:
        print(f"❌ Authentication Error: Invalid API key")
        print(f"Details: {e}")
        return False
        
    except litellm.exceptions.RateLimitError as e:
        print(f"⚠️  Rate Limit Error: Too many requests")
        print(f"Details: {e}")
        return False
        
    except litellm.exceptions.APIConnectionError as e:
        print(f"❌ Connection Error: Cannot reach Mistral servers")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("- Network connectivity issues")
        print("- Mistral API servers are down")
        print("- Firewall blocking the connection")
        return False
        
    except litellm.exceptions.Timeout as e:
        print(f"⏱️  Timeout Error: Request took too long")
        print(f"Details: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}")
        print(f"Details: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  MISTRAL API CONNECTION TEST")
    print("=" * 60)
    print()
    
    success = test_mistral_connection()
    
    print()
    print("=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Ready to run stock analysis!")
    else:
        print("❌ TEST FAILED - Please check the error messages above")
    print("=" * 60)
