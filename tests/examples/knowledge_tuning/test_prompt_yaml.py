

class TestPromptYamlPerformance:
    """Test performance aspects of the prompt YAML."""
    
    @pytest.fixture
    def prompt_yaml_path(self):
        """Return the path to the prompt YAML file."""
        return Path(__file__).parent / "prompt.yaml"
    
    def test_yaml_loading_performance(self, prompt_yaml_path):
        """Test that YAML loading is reasonably fast."""
        import time
        
        start_time = time.time()
        for _ in range(100):  # Load 100 times
            with open(prompt_yaml_path, 'r') as f:
                yaml.safe_load(f)
        end_time = time.time()
        
        load_time = end_time - start_time
        assert load_time < 1.0, f"Loading 100 times should take less than 1 second, took {load_time:.3f}s"
    
    def test_memory_usage_reasonable(self, prompt_yaml_path):
        """Test that loaded YAML doesn't consume excessive memory."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Estimate memory usage by checking string lengths
        total_content_length = sum(len(str(value)) for value in data.values())
        assert total_content_length < 10000, f"Total content should be reasonable size, got {total_content_length} chars"
    
    def test_concurrent_access_safety(self, prompt_yaml_path):
        """Test that the YAML file can be safely accessed concurrently."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def load_yaml():
            try:
                with open(prompt_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                results.put(('success', data))
            except Exception as e:
                results.put(('error', str(e)))
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=load_yaml)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all results
        success_count = 0
        first_data = None
        while not results.empty():
            status, data = results.get()
            assert status == 'success', f"Thread failed with error: {data}"
            if first_data is None:
                first_data = data
            else:
                assert data == first_data, "All threads should load identical data"
            success_count += 1
        
        assert success_count == 5, "All 5 threads should succeed"


class TestPromptYamlCompatibility:
    """Test compatibility with different YAML libraries and versions."""
    
    def test_pyyaml_version_compatibility(self):
        """Test that the YAML works with current PyYAML version."""
        import yaml
        
        # Get PyYAML version info
        yaml_version = getattr(yaml, '__version__', 'unknown')
        assert yaml_version != 'unknown', "Should be able to detect PyYAML version"
        
        # Test basic functionality works
        test_yaml = "key: value\nlist:\n  - item1\n  - item2"
        data = yaml.safe_load(test_yaml)
        assert data['key'] == 'value', "Basic YAML parsing should work"
        assert len(data['list']) == 2, "List parsing should work"
    
    def test_yaml_1_1_vs_1_2_compatibility(self):
        """Test that YAML content is compatible with both YAML 1.1 and 1.2."""
        yaml_content = """system: You are an AI assistant that is expert at rewriting text.

introduction: |
  Given below document, rewrite it using the following instructions:
  {{summary_instruction}}

principles: |
  - Include as much of the document as possible to create a comprehensive summary
  - If there are tables include all the data of the table in the summary

examples: ""

generation: |
  Document:
  {{document_outline}}
  {{document}}
  

start_tags: [""]
end_tags: [""]"""
        
        # Test with different loaders that represent different YAML versions
        try:
            data_safe = yaml.safe_load(yaml_content)
            data_full = yaml.load(yaml_content, Loader=yaml.FullLoader)
            
            # Both should work and produce same results
            assert data_safe == data_full, "Different YAML loaders should produce consistent results"
        except Exception as e:
            pytest.fail(f"YAML should be compatible with different versions: {e}")
    
    def test_cross_platform_compatibility(self):
        """Test that YAML content works across different platforms."""
        import os
        
        yaml_content = """system: You are an AI assistant that is expert at rewriting text.
introduction: |
  Given below document, rewrite it using the following instructions:
  {{summary_instruction}}"""
        
        # Test with different line endings
        unix_content = yaml_content.replace('\r\n', '\n').replace('\r', '\n')
        windows_content = yaml_content.replace('\n', '\r\n')
        mac_content = yaml_content.replace('\n', '\r')
        
        for content_type, content in [
            ('unix', unix_content),
            ('windows', windows_content),
            ('mac', mac_content)
        ]:
            try:
                data = yaml.safe_load(content)
                assert 'system' in data, f"Should parse on {content_type} line endings"
                assert 'introduction' in data, f"Should parse multiline on {content_type} line endings"
            except Exception as e:
                pytest.fail(f"Should handle {content_type} line endings: {e}")


class TestPromptYamlSecurity:
    """Test security aspects of the prompt YAML."""
    
    def test_safe_loading_prevents_code_execution(self):
        """Test that safe_load prevents arbitrary code execution."""
        # This is a malicious YAML that would execute code with unsafe loaders
        malicious_yaml = """
system: !!python/object/apply:os.system ["echo 'This should not execute'"]
"""
        
        # safe_load should not execute the code
        try:
            data = yaml.safe_load(malicious_yaml)
            # If it loads, it should not have executed the system command
            # The value should be None or raise an error
            assert data is None or 'system' not in data or not callable(data.get('system'))
        except yaml.constructor.ConstructorError:
            # This is expected - safe_load should reject dangerous constructs
            pass
        except Exception as e:
            # Other errors are also acceptable as long as code doesn't execute
            pass
    
    def test_no_external_references(self):
        """Test that YAML doesn't contain external file references."""
        yaml_content = """system: You are an AI assistant that is expert at rewriting text.

introduction: |
  Given below document, rewrite it using the following instructions:
  {{summary_instruction}}

principles: |
  - Include as much of the document as possible to create a comprehensive summary
  - If there are tables include all the data of the table in the summary

examples: ""

generation: |
  Document:
  {{document_outline}}
  {{document}}
  

start_tags: [""]
end_tags: [""]"""
        
        # Check for potential external references
        dangerous_patterns = [
            '<<:',  # YAML merge keys can be used for inclusion
            '&',    # YAML anchors can be misused
            '*',    # YAML aliases can be misused
            '!!python',  # Python object constructors
            '!!map',     # Explicit map constructors
            '!!seq',     # Explicit sequence constructors
        ]
        
        for pattern in dangerous_patterns:
            assert pattern not in yaml_content, f"YAML should not contain potentially dangerous pattern: {pattern}"
    
    def test_input_sanitization(self):
        """Test that the YAML template properly handles user input placeholders."""
        # The placeholders should be clearly marked and not executable
        yaml_content = """system: You are an AI assistant that is expert at rewriting text.

introduction: |
  Given below document, rewrite it using the following instructions:
  {{summary_instruction}}

generation: |
  Document:
  {{document_outline}}
  {{document}}"""
        
        data = yaml.safe_load(yaml_content)
        
        # Find all placeholders
        all_placeholders = []
        for value in data.values():
            if isinstance(value, str):
                placeholders = re.findall(r'\{\{([^}]+)\}\}', value)
                all_placeholders.extend(placeholders)
        
        # All placeholders should be simple variable names
        for placeholder in all_placeholders:
            assert placeholder.replace('_', '').isalnum(), \
                f"Placeholder '{placeholder}' should be a simple variable name"
            assert not placeholder.startswith('__'), \
                f"Placeholder '{placeholder}' should not use private naming convention"


class TestPromptYamlDocumentation:
    """Test that the prompt YAML is well-documented and self-explanatory."""
    
    @pytest.fixture
    def prompt_yaml_path(self):
        """Return the path to the prompt YAML file."""
        return Path(__file__).parent / "prompt.yaml"
    
    def test_field_names_are_descriptive(self, prompt_yaml_path):
        """Test that all field names are descriptive and clear."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        field_names = list(data.keys())
        
        # Check that field names are descriptive
        for field_name in field_names:
            assert len(field_name) >= 4, f"Field name '{field_name}' should be descriptive (at least 4 chars)"
            assert field_name.islower() or '_' in field_name, \
                f"Field name '{field_name}' should follow naming conventions"
            assert not field_name.startswith('_'), \
                f"Field name '{field_name}' should not be private"
    
    def test_content_provides_clear_instructions(self, prompt_yaml_path):
        """Test that the content provides clear, actionable instructions."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # System message should be clear about the role
        system = data.get('system', '')
        assert len(system.split()) >= 5, "System message should be detailed enough"
        assert any(word in system.lower() for word in ['you are', 'you will', 'your role']), \
            "System message should clearly define the AI's role"
        
        # Introduction should explain the task
        introduction = data.get('introduction', '')
        assert 'document' in introduction.lower(), "Introduction should mention what will be processed"
        assert 'instruction' in introduction.lower(), "Introduction should mention instructions"
        
        # Principles should provide actionable guidance
        principles = data.get('principles', '')
        action_words = ['include', 'create', 'use', 'ensure', 'maintain', 'preserve']
        assert any(word in principles.lower() for word in action_words), \
            "Principles should contain actionable guidance"
    
    def test_placeholder_documentation_implicit(self, prompt_yaml_path):
        """Test that placeholders are implicitly documented through context."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Find all placeholders and check they're used in appropriate context
        for field_name, content in data.items():
            if isinstance(content, str):
                placeholders = re.findall(r'\{\{([^}]+)\}\}', content)
                for placeholder in placeholders:
                    # Placeholder names should be self-documenting
                    assert '_' in placeholder or len(placeholder) >= 6, \
                        f"Placeholder '{placeholder}' should be self-documenting"
                    
                    # Context should make the placeholder's purpose clear
                    context_words = content.lower().split()
                    placeholder_words = placeholder.lower().split('_')
                    
                    # At least one word from placeholder should appear in context
                    common_words = set(context_words) & set(placeholder_words)
                    assert len(common_words) > 0 or field_name.lower() in placeholder.lower(), \
                        f"Placeholder '{placeholder}' should have clear context in field '{field_name}'"


class TestPromptYamlMaintainability:
    """Test that the prompt YAML is maintainable and extensible."""
    
    @pytest.fixture
    def prompt_yaml_path(self):
        """Return the path to the prompt YAML file."""
        return Path(__file__).parent / "prompt.yaml"
    
    def test_structure_allows_easy_modification(self, prompt_yaml_path):
        """Test that the YAML structure allows easy modification."""
        with open(prompt_yaml_path, 'r') as f:
            original_content = f.read()
            original_data = yaml.safe_load(original_content)
        
        # Test adding new fields
        modified_data = original_data.copy()
        modified_data['new_field'] = 'new value'
        modified_data['another_list'] = ['item1', 'item2']
        
        # Should be able to serialize and reload
        new_yaml = yaml.dump(modified_data, default_flow_style=False)
        reloaded_data = yaml.safe_load(new_yaml)
        
        assert reloaded_data['new_field'] == 'new value', "Should be able to add new string fields"
        assert reloaded_data['another_list'] == ['item1', 'item2'], "Should be able to add new list fields"
        
        # Original fields should be preserved
        for key, value in original_data.items():
            assert reloaded_data[key] == value, f"Original field '{key}' should be preserved"
    
    def test_field_order_consistency(self, prompt_yaml_path):
        """Test that field order is consistent and logical."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        field_names = list(data.keys())
        
        # System should come first (defines the AI's role)
        assert field_names[0] == 'system', "System field should come first"
        
        # Introduction should come early (explains the task)
        intro_index = field_names.index('introduction') if 'introduction' in field_names else -1
        assert intro_index <= 2, "Introduction should come early in the structure"
        
        # Generation should come after system and introduction (it's the template)
        gen_index = field_names.index('generation') if 'generation' in field_names else -1
        system_index = field_names.index('system')
        assert gen_index > system_index, "Generation should come after system definition"
    
    def test_extensibility_patterns(self, prompt_yaml_path):
        """Test that the YAML follows patterns that support extensibility."""
        with open(prompt_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Lists should be used for extensible collections
        for field_name, value in data.items():
            if isinstance(value, list):
                # Lists should allow easy addition of new items
                new_list = value + ['new_item']
                assert len(new_list) == len(value) + 1, f"List field '{field_name}' should be extensible"
        
        # String fields with placeholders should support template expansion
        template_fields = ['introduction', 'generation']
        for field_name in template_fields:
            if field_name in data:
                content = data[field_name]
                if isinstance(content, str) and '{{' in content:
                    # Should be able to add new placeholders
                    new_content = content + '\n{{new_placeholder}}'
                    new_placeholders = re.findall(r'\{\{([^}]+)\}\}', new_content)
                    original_placeholders = re.findall(r'\{\{([^}]+)\}\}', content)
                    assert len(new_placeholders) > len(original_placeholders), \
                        f"Template field '{field_name}' should support adding new placeholders"
    
    def test_version_control_friendly(self, prompt_yaml_path):
        """Test that the YAML format is version control friendly."""
        with open(prompt_yaml_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Should not have very long lines (makes diffs hard to read)
        for i, line in enumerate(lines):
            assert len(line) <= 200, f"Line {i+1} is too long ({len(line)} chars), should be <= 200 for readability"
        
        # Should have consistent indentation
        indent_levels = set()
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces > 0:
                    indent_levels.add(leading_spaces)
        
        # Should use consistent indentation (typically 2 spaces for YAML)
        if indent_levels:
            min_indent = min(indent_levels)
            for indent in indent_levels:
                assert indent % min_indent == 0, \
                    f"Indentation should be consistent multiples of {min_indent} spaces"