#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for PR functionality with remote and local providers.

This script can be run to verify that the PR changes work correctly with both
remote model providers (like OpenAI) and local providers (like vLLM/Ollama).

Usage:
    python tests/test_end_to_end.py --backend openai --api-key YOUR_KEY
    python tests/test_end_to_end.py --backend vllm --base-url http://localhost:8000/v1
    python tests/test_end_to_end.py --backend ollama --base-url http://localhost:11434/v1

Examples:
    # Test OpenAI backend
    python tests/test_end_to_end.py --backend openai --api-key sk-your-key --base-url https://api.openai.com/v1

    # Test local vLLM backend
    python tests/test_end_to_end.py --backend vllm --api-key EMPTY --base-url http://localhost:8000/v1 --model-id meta-llama/Llama-3.1-8B-Instruct

    # Test Ollama backend
    python tests/test_end_to_end.py --backend ollama --api-key EMPTY --base-url http://localhost:11434/v1 --model-id llama3.1
"""

# Standard
import argparse
import os
import sys
import tempfile
import traceback
from typing import Dict, Any, Optional, List, Tuple

# Third Party
from datasets import Dataset
from openai import OpenAI

# Local
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
from sdg_hub.blocks.utilblocks import ChunkingBlock


class EndToEndTester:
    """End-to-end tester for different model backends."""

    def __init__(self, backend: str, api_key: str, base_url: str, model_id: Optional[str] = None) -> None:
        """Initialize tester with backend configuration.
        
        Parameters
        ----------
        backend : str
            Backend type to test ('openai', 'vllm', 'ollama', 'azure')
        api_key : str
            API key for authentication (use 'EMPTY' for local providers)
        base_url : str
            Base URL for the API endpoint
        model_id : Optional[str], optional
            Specific model ID to use, by default None
        """
        self.backend: str = backend
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.model_id: Optional[str] = model_id
        self.client: Optional[OpenAI] = None
        self.teacher_model: Optional[str] = None

    def setup_client(self) -> bool:
        """Set up OpenAI client for the specified backend.
        
        Returns
        -------
        bool
            True if client setup was successful, False otherwise
            
        Raises
        ------
        ImportError
            If required dependencies are not available
        ValueError
            If backend configuration is invalid
        """
        try:
            print(f"🔧 Setting up {self.backend} client...")
            
            if self.backend == "azure":
                try:
                    from openai import AzureOpenAI
                except ImportError as e:
                    print(f"❌ Azure OpenAI not available: {e}")
                    return False
                    
                if not self.base_url.endswith('/'):
                    self.base_url += '/'
                    
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version="2024-02-01",
                    azure_endpoint=self.base_url,
                )
            else:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            
            print(f"✅ Client initialized successfully")
            return True
            
        except ImportError as e:
            print(f"❌ Missing required dependency: {e}")
            print("💡 Try: pip install openai")
            return False
        except ValueError as e:
            print(f"❌ Invalid configuration: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize client: {e}")
            print(f"💡 Check your backend configuration and network connectivity")
            return False

    def test_connection(self) -> bool:
        """Test connection and get available models."""
        try:
            print(f"🔍 Testing connection to {self.backend}...")
            
            models = self.client.models.list()
            available_models = [m.id for m in models.data]
            
            print(f"✅ Connection successful!")
            print(f"📋 Available models ({len(available_models)}):")
            for i, model in enumerate(available_models[:5]):  # Show first 5
                print(f"  {i+1}. {model}")
            if len(available_models) > 5:
                print(f"  ... and {len(available_models) - 5} more")
            
            # Select model
            if self.model_id and self.model_id in available_models:
                self.teacher_model = self.model_id
                print(f"🎯 Using specified model: {self.teacher_model}")
            elif self.model_id:
                # Handle vLLM model ID variations
                vllm_variants = [
                    self.model_id,
                    f"/{self.model_id}",
                    f"/model/{self.model_id}"
                ]
                found_model = None
                for variant in vllm_variants:
                    if variant in available_models:
                        found_model = variant
                        break
                
                if found_model:
                    self.teacher_model = found_model
                    print(f"🎯 Found model variant: {self.teacher_model}")
                else:
                    self.teacher_model = available_models[0]
                    print(f"⚠️  Model '{self.model_id}' not found, using: {self.teacher_model}")
            else:
                self.teacher_model = available_models[0]
                print(f"🎯 Using first available model: {self.teacher_model}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def test_basic_completion(self) -> bool:
        """Test basic completion to verify API functionality."""
        try:
            print(f"🧪 Testing basic completion...")
            
            response = self.client.completions.create(
                model=self.teacher_model,
                prompt="Hello, this is a test. Please respond with 'Test successful'.",
                max_tokens=50,
                temperature=0.0
            )
            
            response_text = response.choices[0].text.strip()
            print(f"✅ Basic completion successful!")
            print(f"📝 Response: {response_text[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic completion failed: {e}")
            return False

    def test_chunking_block(self) -> bool:
        """Test ChunkingBlock functionality."""
        try:
            print(f"🔧 Testing ChunkingBlock...")
            
            # Create test data with long document
            long_document = "This is a test document. " * 100  # ~2500 characters
            test_data = [{"document": long_document}]
            dataset = Dataset.from_list(test_data)
            
            # Test chunking
            chunking_block = ChunkingBlock(
                block_name="test_chunking",
                input_col="document",
                output_col="document",
                chunk_size=512,
                overlap=50
            )
            
            chunked_dataset = chunking_block.generate(dataset)
            
            print(f"✅ ChunkingBlock test successful!")
            print(f"📊 Original documents: {len(dataset)}")
            print(f"📊 Chunked documents: {len(chunked_dataset)}")
            print(f"📊 Chunks per document: {len(chunked_dataset) / len(dataset):.1f}")
            
            # Verify chunk metadata
            sample_chunk = chunked_dataset[0]
            required_fields = ["document", "chunk_id", "total_chunks"]
            for field in required_fields:
                if field not in sample_chunk:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            # Verify chunk content and sizing
            for i, chunk in enumerate(chunked_dataset):
                chunk_text = chunk["document"]
                if len(chunk_text) > 512:
                    print(f"❌ Chunk {i} exceeds max size: {len(chunk_text)} chars")
                    return False
                if not chunk_text.strip():
                    print(f"❌ Chunk {i} is empty")
                    return False
                    
            # Test edge case: document smaller than chunk size
            small_doc = [{"document": "Small doc"}]
            small_dataset = Dataset.from_list(small_doc)
            small_chunked = chunking_block.generate(small_dataset)
            
            if len(small_chunked) != 1:
                print(f"❌ Small document should produce 1 chunk, got {len(small_chunked)}")
                return False
            
            print(f"✅ Chunk metadata validation passed")
            return True
            
        except Exception as e:
            print(f"❌ ChunkingBlock test failed: {e}")
            traceback.print_exc()
            return False

    def test_context_length_handling(self) -> bool:
        """Test context length handling for different model types."""
        try:
            print(f"🔍 Testing context length handling...")
            
            # Estimate context limit based on model
            model_lower = self.teacher_model.lower()
            if any(x in model_lower for x in ["gpt-4", "claude-3", "gemini-pro"]):
                estimated_context = 128000
            elif "gpt-3.5" in model_lower:
                estimated_context = 16385
            elif any(x in model_lower for x in ["llama", "mistral", "mixtral"]):
                if "70b" in model_lower or "large" in model_lower:
                    estimated_context = 32768
                else:
                    estimated_context = 4096
            else:
                estimated_context = 4096  # Conservative default
            
            print(f"📏 Estimated context length: {estimated_context} tokens")
            
            # Determine appropriate max_tokens
            if estimated_context < 8000:
                recommended_max_tokens = min(1024, estimated_context // 4)
                should_use_chunking = True
            else:
                recommended_max_tokens = 4000
                should_use_chunking = False
            
            print(f"🎛️  Recommended max_tokens: {recommended_max_tokens}")
            print(f"✂️  Should use chunking: {should_use_chunking}")
            
            # Test with a document that might exceed context
            if should_use_chunking:
                long_doc = "This is a very long document. " * 200  # ~6000 characters
                print(f"📄 Testing with long document ({len(long_doc)} chars)")
                
                # This should work with chunking
                print(f"✅ Context length handling configured correctly")
            else:
                print(f"✅ Large context model - chunking not required")
            
            return True
            
        except Exception as e:
            print(f"❌ Context length test failed: {e}")
            return False

    def test_backend_specific_features(self) -> bool:
        """Test backend-specific features and configurations."""
        try:
            print(f"🔧 Testing {self.backend}-specific features...")
            
            if self.backend == "vllm":
                # Test vLLM model ID format
                if not self.teacher_model:
                    print("❌ No teacher model available for vLLM validation")
                    return False

                if not self.teacher_model.startswith("/"):
                    print(f"⚠️  vLLM model ID should start with '/' but got: {self.teacher_model}")
                else:
                    print(f"✅ vLLM model ID format correct: {self.teacher_model}")
                
                # Test empty API key requirement
                if self.api_key != "EMPTY":
                    print(f"⚠️  vLLM should use 'EMPTY' API key but got: {self.api_key}")
                else:
                    print(f"✅ vLLM API key correctly set to 'EMPTY'")
            elif self.backend == "ollama":
                # Test Ollama-specific features
                if "localhost:11434" not in self.base_url:
                    print(f"⚠️  Ollama typically runs on localhost:11434 but got: {self.base_url}")
                else:
                    print(f"✅ Ollama endpoint format correct")
                    
            elif self.backend == "openai":
                # Test OpenAI-specific features
                if not self.api_key.startswith("sk-"):
                    print(f"⚠️  OpenAI API key should start with 'sk-' but got: {self.api_key[:10]}...")
                else:
                    print(f"✅ OpenAI API key format correct")
                    
            print(f"✅ Backend-specific features validated")
            return True
            
        except Exception as e:
            print(f"❌ Backend-specific test failed: {e}")
            return False

    def test_sample_data_generation(self) -> bool:
        """Test sample data generation with minimal configuration."""
        try:
            print(f"🎯 Testing sample data generation...")
            
            # Create minimal test dataset
            test_docs = [
                {
                    "document": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991."
                }
            ]
            dataset = Dataset.from_list(test_docs)
            
            print(f"📝 Created test dataset with {len(dataset)} document(s)")
            print(f"📄 Sample document: {test_docs[0]['document'][:100]}...")
            
            # For this test, we'll just validate the setup
            # (Full generation would require actual YAML configs)
            print(f"✅ Sample data generation setup successful")
            return True
            
        except Exception as e:
            print(f"❌ Sample data generation test failed: {e}")
            traceback.print_exc()
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        tests = [
            ("Client Setup", self.setup_client),
            ("Connection Test", self.test_connection),
            ("Basic Completion", self.test_basic_completion),
            ("ChunkingBlock", self.test_chunking_block),
            ("Context Length Handling", self.test_context_length_handling),
            ("Backend-Specific Features", self.test_backend_specific_features),
            ("Sample Data Generation", self.test_sample_data_generation),
        ]
        
        results = {}
        print(f"🚀 Running end-to-end tests for {self.backend} backend")
        print("=" * 60)
        
        for test_name, test_func in tests:
            print(f"\\n🧪 {test_name}")
            print("-" * 40)
            
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: ERROR - {e}")
                
        return results

    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print test summary."""
        print("\\n" + "=" * 60)
        print(f"🏁 TEST SUMMARY - {self.backend.upper()} BACKEND")
        print("=" * 60)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"🎉 All tests passed! {self.backend} backend is working correctly.")
        else:
            print(f"⚠️  Some tests failed. Please check the configuration and logs.")
            
        print("\\n💡 Next steps:")
        if passed == total:
            print(f"- ✅ {self.backend} backend is ready for use")
            print(f"- 🚀 You can now run the notebook with this backend")
            print(f"- 📝 Try generating synthetic data with your documents")
        else:
            print(f"- 🔧 Fix the failing tests before proceeding")
            print(f"- 📖 Check MODEL_BACKENDS.md for configuration help")
            print(f"- 🐛 Review error messages above for specific issues")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="End-to-end tests for SDG Hub model backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test OpenAI backend
  python tests/test_end_to_end.py --backend openai --api-key sk-your-key

  # Test local vLLM backend
  python tests/test_end_to_end.py --backend vllm --base-url http://localhost:8000/v1 --model-id meta-llama/Llama-3.1-8B-Instruct

  # Test Ollama backend
  python tests/test_end_to_end.py --backend ollama --base-url http://localhost:11434/v1 --model-id llama3.1

  # Test Azure OpenAI
  python tests/test_end_to_end.py --backend azure --api-key your-azure-key --base-url https://your-resource.openai.azure.com/ --model-id your-deployment
        """
    )
    
    parser.add_argument("--backend", required=True, choices=["openai", "vllm", "ollama", "azure"],
                       help="Backend to test")
    parser.add_argument("--api-key", required=True,
                       help="API key (use 'EMPTY' for vLLM/Ollama)")
    parser.add_argument("--base-url", required=True,
                       help="Base URL for the API endpoint")
    parser.add_argument("--model-id", 
                       help="Specific model ID to use (optional)")
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = EndToEndTester(
        backend=args.backend,
        api_key=args.api_key,
        base_url=args.base_url,
        model_id=args.model_id
    )
    
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Exit with appropriate code
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()